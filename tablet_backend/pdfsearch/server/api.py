"""api-spec v1 HTTP 서버 (Flutter 앱 lib/api/api_client.dart 와 짝).

    POST /api/v1/documents                          PDF 업로드 -> 202 processing (같은 PDF 면 200)
    GET  /api/v1/documents                          등록된 문서 목록 (용량 포함)
    GET  /api/v1/documents/{document_id}            처리 상태
    DELETE /api/v1/documents/{document_id}          문서 삭제 (그림·임베딩 폴더째)
    POST /api/v1/documents/{document_id}/search     스케치 PNG -> 유사 그림 top_k
    GET  /api/v1/documents/{document_id}/figures/{figure_id}/image?size=thumb|full
    GET  /api/v1/health

에러 응답: {"error": {"code": "...", "message": "..."}}

실행 (tablet_backend/ 에서):
    uvicorn pdfsearch.server.api:app --host 0.0.0.0 --port 8001
환경변수:
    PDFSEARCH_DATA            문서 저장 폴더 (기본 tablet_backend/server_data)
    PDFSEARCH_MODE            검색 모드 (기본 edge, 임베딩을 edge 만 저장하므로 바꾸지 말 것)
    PDFSEARCH_PORT            앱에 알려 줄 HTTP 포트 (기본 8001, uvicorn --port 와 같게)
    PDFSEARCH_DISCOVERY_PORT  자동 찾기 UDP 포트 (기본 8002, 앱과 같아야 함)
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from pdfsearch.server.discovery import DiscoveryResponder, lan_addresses
from pdfsearch.server.store import BASE_DIR, DocumentStore, Engines, default_config

log = logging.getLogger("server.api")

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_SKETCH_BYTES = 10 * 1024 * 1024
THUMB_LONG_EDGE = 512
INK_TH = 200                    # features.TRIM_TH 와 같은 기준: 이보다 어두우면 선


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def _bbox_xywh(rec: dict) -> list[float]:
    """figures.jsonl 의 xyxy_px(렌더 이미지 픽셀) -> [x, y, w, h] 0~1, 좌상단 원점."""
    dbg = rec.get("debug") or {}
    xyxy, size = dbg.get("xyxy_px"), dbg.get("size_px")
    if not xyxy or not size:
        return [0.0, 0.0, 1.0, 1.0]
    (x0, y0, x1, y1), (W, H) = xyxy, size
    return [round(x0 / W, 6), round(y0 / H, 6), round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)]


def _doc_json(st: dict) -> dict:
    out = {k: st.get(k) for k in ("document_id", "filename", "status", "page_count", "figure_count")}
    if st.get("error"):
        out["error"] = st["error"]
    return out


def create_app(data_dir: Path | None = None, engines: Engines | None = None,
               mode: str | None = None, discovery: bool = True) -> FastAPI:
    data_dir = Path(data_dir or os.environ.get("PDFSEARCH_DATA") or BASE_DIR / "server_data")
    mode = mode or os.environ.get("PDFSEARCH_MODE", "edge")
    engines = engines or Engines(default_config())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = DocumentStore(data_dir, engines)
        log.info("데이터 폴더: %s, 검색 모드: %s", data_dir, mode)
        http_port = int(os.environ.get("PDFSEARCH_PORT", "8001"))
        responder = None
        if discovery:
            responder = DiscoveryResponder(
                http_port, int(os.environ.get("PDFSEARCH_DISCOVERY_PORT", "8002")))
            responder.start()
        for ip in lan_addresses():
            log.info("앱 직접 입력용 주소: http://%s:%d", ip, http_port)
        yield
        if responder:
            responder.stop()
        app.state.store.shutdown()

    app = FastAPI(title="PDF 그림 스케치 검색", version="1", lifespan=lifespan)

    # Flutter 를 Chrome(웹)으로 띄우면 앱 주소(localhost:xxxxx)와 서버 주소(:8001)가 달라
    # 브라우저가 응답을 막는다(CORS). 로그인/쿠키를 쓰지 않으므로 모든 출처 허용.
    # 네이티브 앱(태블릿, Windows)은 CORS 와 무관.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    def store(request: Request) -> DocumentStore:
        return request.app.state.store

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.status)

    @app.exception_handler(Exception)
    async def _unknown(_: Request, e: Exception):
        log.exception("처리되지 않은 오류")
        return JSONResponse({"error": {"code": "INTERNAL_ERROR", "message": "서버 오류가 발생했습니다."}},
                            status_code=500)

    def _ready_doc(request: Request, document_id: str) -> dict:
        st = store(request).get(document_id)
        if st is None:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "문서를 찾을 수 없습니다.")
        if st["status"] == "processing":
            raise ApiError(409, "DOCUMENT_NOT_READY", "아직 PDF를 처리하고 있습니다.")
        if st["status"] == "failed":
            err = st.get("error") or {}
            raise ApiError(409, "DOCUMENT_NOT_READY", err.get("message", "PDF 처리에 실패한 문서입니다."))
        return st

    # ------------------------------------------------------------ 엔드포인트
    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "mode": mode}

    @app.post("/api/v1/documents")
    async def upload(request: Request, file: UploadFile = File(...)):
        data = await file.read(MAX_PDF_BYTES + 1)
        if len(data) > MAX_PDF_BYTES:
            raise ApiError(413, "FILE_TOO_LARGE", "파일이 너무 큽니다. (최대 50MB)")
        if not data.startswith(b"%PDF"):
            raise ApiError(400, "INVALID_FILE", "PDF 파일이 아니거나 손상되었습니다.")
        name = Path(file.filename or "document.pdf").name
        st, started = store(request).submit(data, name)
        return JSONResponse(_doc_json(st), status_code=202 if started else 200)

    @app.get("/api/v1/documents")
    def list_documents(request: Request):
        docs = []
        for st in store(request).list():
            d = _doc_json(st)
            d["size_bytes"] = st["size_bytes"]
            d["created_at"] = st["created_at"]
            docs.append(d)
        return {"documents": docs}

    @app.delete("/api/v1/documents/{document_id}")
    def delete_document(request: Request, document_id: str):
        try:
            store(request).delete(document_id)
        except KeyError:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "문서를 찾을 수 없습니다.")
        except RuntimeError:
            raise ApiError(409, "DOCUMENT_NOT_READY", "처리 중인 문서는 지울 수 없습니다.")
        return {"document_id": document_id, "deleted": True}

    @app.get("/api/v1/documents/{document_id}")
    def get_document(request: Request, document_id: str):
        st = store(request).get(document_id)
        if st is None:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "문서를 찾을 수 없습니다.")
        return _doc_json(st)

    @app.post("/api/v1/documents/{document_id}/search")
    async def search(request: Request, document_id: str,
                     sketch: UploadFile = File(...), top_k: int = Form(3)):
        st = _ready_doc(request, document_id)
        if not st.get("searchable_count"):
            raise ApiError(422, "NO_FIGURES", "이 PDF에서 그림을 찾지 못했습니다.")

        raw = await sketch.read(MAX_SKETCH_BYTES + 1)
        if len(raw) > MAX_SKETCH_BYTES:
            raise ApiError(413, "FILE_TOO_LARGE", "스케치 이미지가 너무 큽니다.")
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
        if bgr is None:
            raise ApiError(400, "INVALID_FILE", "스케치 이미지를 읽을 수 없습니다.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < INK_TH).any():
            raise ApiError(400, "EMPTY_SKETCH", "먼저 그림을 그려 주세요.")

        k = max(1, min(int(top_k), 10))
        s = store(request)
        figs = s.figures(document_id)
        results = []
        for r in s.search(document_id, rgb, k, mode):
            rec = figs.get(r["fig_id"], {})
            results.append({
                "figure_id": r["fig_id"],
                "score": round(max(0.0, min(1.0, r["score"])), 4),
                "page": r["page"],
                "bbox": _bbox_xywh(rec),
                "image_url": f"/api/v1/documents/{document_id}/figures/{r['fig_id']}/image?size=thumb",
                "caption": None,
                "level": r.get("level", "figure"),
            })
        return {"document_id": document_id, "results": results}

    @app.get("/api/v1/documents/{document_id}/figures/{figure_id}/image")
    def figure_image(request: Request, document_id: str, figure_id: str, size: str = "thumb"):
        s = store(request)
        if s.get(document_id) is None:
            raise ApiError(404, "DOCUMENT_NOT_FOUND", "문서를 찾을 수 없습니다.")
        rec = s.figures(document_id).get(figure_id)
        path = s.doc_dir(document_id) / rec["file"] if rec else None
        if path is None or not path.exists():
            raise ApiError(404, "FIGURE_NOT_FOUND", "그림을 찾을 수 없습니다.")
        headers = {"Cache-Control": "public, max-age=86400"}
        if size != "thumb":
            return FileResponse(path, media_type="image/png", headers=headers)
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        scale = THUMB_LONG_EDGE / max(h, w)
        if scale < 1:
            img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", img)
        return Response(buf.tobytes(), media_type="image/png", headers=headers)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
