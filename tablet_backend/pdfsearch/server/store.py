"""업로드된 PDF 문서 관리 + 백그라운드 처리.

문서 1개 = 폴더 1개 (기존 dataset/ 과 같은 구조라 FigureSearcher 를 그대로 쓴다):
    server_data/documents/<document_id>/
        source.pdf
        status.json            {"document_id","filename","status","page_count","figure_count","error"}
        figures/*.png
        features/{figures.jsonl, meta.jsonl, embeddings_*.npy, model.json, ...}

document_id = PDF 내용의 sha256 앞 16자리 -> 같은 PDF 를 다시 올리면 처리 없이 바로 ready.
처리(검출+임베딩)는 작업 스레드 1개에서 순서대로 실행한다 (CPU/메모리 보호).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from pdfsearch.common.config import Config
from pdfsearch.ingest.saver import atomic_write_text, read_jsonl

log = logging.getLogger("server.store")

BASE_DIR = Path(__file__).resolve().parents[2]          # tablet_backend/


class Engines:
    """무거운 모델(검출기, DINOv2)을 한 번만 만들어 공유. 테스트에서는 가짜로 바꿔 끼운다."""

    def __init__(self, cfg: Config,
                 detector_factory: Callable[[Config], object] | None = None,
                 embedder_factory: Callable[[], object] | None = None):
        self.cfg = cfg
        self._detector_factory = detector_factory
        self._embedder_factory = embedder_factory
        self._detector = None
        self._embedder = None
        self._load_lock = threading.Lock()
        # torch 모델을 여러 스레드가 동시에 돌리지 않게
        self.embed_lock = threading.Lock()

    @property
    def detector(self):
        with self._load_lock:
            if self._detector is None:
                if self._detector_factory:
                    self._detector = self._detector_factory(self.cfg)
                else:
                    from pdfsearch.ingest.detector_yolo import FigureDetector
                    self._detector = FigureDetector(self.cfg)
            return self._detector

    @property
    def embedder(self):
        with self._load_lock:
            if self._embedder is None:
                if self._embedder_factory:
                    self._embedder = self._embedder_factory()
                else:
                    import warnings
                    from pdfsearch.embedding.features import Embedder
                    # DINOv2 는 GPU 가속 라이브러리 xFormers 가 없으면 경고를 낸다.
                    # CPU 에서는 원래 쓰지 않으므로 무시해도 결과는 같다.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="xFormers is not available")
                        self._embedder = Embedder()
            return self._embedder


class _LockedEmbedder:
    """build_features / FigureSearcher 에 넘길 때 embed_lock 을 거는 얇은 래퍼."""

    def __init__(self, engines: Engines):
        self._e = engines

    @property
    def dim(self):
        return self._e.embedder.dim

    def embed_images(self, rgbs, mode, batch: int = 16):
        with self._e.embed_lock:
            return self._e.embedder.embed_images(rgbs, mode, batch)


class DocumentStore:
    def __init__(self, data_dir: Path, engines: Engines):
        self.root = Path(data_dir) / "documents"
        self.root.mkdir(parents=True, exist_ok=True)
        self.engines = engines
        self._lock = threading.Lock()
        self._status: dict[str, dict] = {}
        self._searchers: dict[str, object] = {}
        self._figs: dict[str, dict[str, dict]] = {}     # doc -> fig_id -> figures.jsonl 레코드
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")
        self._load_existing()

    # ------------------------------------------------------------ 상태
    def _load_existing(self) -> None:
        for d in self.root.iterdir():
            f = d / "status.json"
            if not f.exists():
                continue
            st = json.loads(f.read_text(encoding="utf-8"))
            if st["status"] == "processing":             # 서버가 처리 중에 꺼졌음
                st["status"] = "failed"
                st["error"] = {"code": "INTERRUPTED",
                               "message": "서버가 다시 시작되어 처리가 중단되었습니다. 다시 업로드해 주세요."}
                self._write_status(st)
            self._status[st["document_id"]] = st
        log.info("기존 문서 %d개 로드", len(self._status))

    def _write_status(self, st: dict) -> None:
        d = self.root / st["document_id"]
        d.mkdir(parents=True, exist_ok=True)
        atomic_write_text(d / "status.json", json.dumps(st, ensure_ascii=False, indent=2))

    def _set(self, doc_id: str, **kw) -> dict:
        with self._lock:
            st = {**self._status[doc_id], **kw}
            self._status[doc_id] = st
            self._write_status(st)
            return dict(st)

    def get(self, doc_id: str) -> dict | None:
        with self._lock:
            st = self._status.get(doc_id)
            return dict(st) if st else None

    def doc_dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    # ------------------------------------------------------------ 업로드
    def submit(self, pdf_bytes: bytes, filename: str) -> tuple[dict, bool]:
        """(상태, 새로 처리 시작했는지). 같은 PDF 가 이미 ready/processing 이면 그대로 돌려준다."""
        doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]
        with self._lock:
            cur = self._status.get(doc_id)
            if cur and cur["status"] in ("ready", "processing"):
                if cur["filename"] != filename:          # 같은 내용, 다른 이름 -> 이름만 갱신
                    cur = {**cur, "filename": filename}
                    self._status[doc_id] = cur
                    self._write_status(cur)
                return dict(cur), False
            d = self.root / doc_id
            d.mkdir(parents=True, exist_ok=True)
            (d / "source.pdf").write_bytes(pdf_bytes)
            st = {"document_id": doc_id, "filename": filename, "status": "processing",
                  "page_count": None, "figure_count": None, "error": None,
                  "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self._status[doc_id] = st
            self._searchers.pop(doc_id, None)
            self._figs.pop(doc_id, None)
            self._write_status(st)
        self._pool.submit(self._ingest, doc_id)
        return dict(st), True

    # ------------------------------------------------------------ 처리 (작업 스레드)
    def _ingest(self, doc_id: str) -> None:
        import shutil
        from pdfsearch.embedding.features import build_features
        from pdfsearch.ingest.pdf_load import iter_pages, page_count
        from pdfsearch.ingest.postprocess import postprocess
        from pdfsearch.ingest.preprocess import preprocess
        from pdfsearch.ingest.saver import Saver

        d = self.doc_dir(doc_id)
        pdf = str(d / "source.pdf")
        try:
            for sub in ("figures", "features"):          # 실패 후 재업로드 대비 초기화
                shutil.rmtree(d / sub, ignore_errors=True)
            try:
                n = page_count(pdf)
            except Exception as e:
                raise _IngestError("INVALID_FILE", "PDF 파일이 아니거나 손상되었습니다.") from e
            self._set(doc_id, page_count=n)

            cfg = replace(self.engines.cfg, output_dir=d)
            det = self.engines.detector
            fails = []
            with Saver(cfg, pdf, getattr(det, "weights_hash", "unknown")) as sv:
                for p in iter_pages(pdf, cfg):
                    try:
                        p = preprocess(p, cfg)
                        sv.save_page(p, postprocess(det.detect(p), p.size_px, cfg))
                    except Exception:
                        log.exception("[%s] page %d 실패 -> 건너뜀", doc_id, p.pdf_index + 1)
                        fails.append(p.pdf_index + 1)
            if fails and len(fails) == n:
                raise _IngestError("INTERNAL_ERROR", "모든 페이지 처리에 실패했습니다.")

            n_search = build_features(d, emb=_LockedEmbedder(self.engines))
            figs = [r for r in read_jsonl(d / "features" / "figures.jsonl")
                    if r.get("level", "figure") == "figure"]
            self._set(doc_id, status="ready", figure_count=len({r["fig_id"] for r in figs}),
                      searchable_count=n_search, failed_pages=fails or None)
            log.info("[%s] ready: %d쪽, 그림 %d개(검색 대상 %d)", doc_id, n, len(figs), n_search)
        except _IngestError as e:
            self._set(doc_id, status="failed", error={"code": e.code, "message": e.message})
        except Exception:
            log.exception("[%s] 처리 실패", doc_id)
            self._set(doc_id, status="failed",
                      error={"code": "INTERNAL_ERROR", "message": "PDF를 처리하지 못했습니다."})

    # ------------------------------------------------------------ 목록 / 삭제
    def list(self) -> list[dict]:
        """등록된 문서 전체 (최근 등록 순). 폴더 용량 포함."""
        with self._lock:
            items = [dict(st) for st in self._status.values()]
        for st in items:
            d = self.doc_dir(st["document_id"])
            st["size_bytes"] = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            if not st.get("created_at"):                 # 예전에 만든 문서: 폴더 시간으로
                st["created_at"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%S", time.localtime((d / "source.pdf").stat().st_mtime)
                    if (d / "source.pdf").exists() else time.localtime())
        return sorted(items, key=lambda s: s["created_at"], reverse=True)

    def delete(self, doc_id: str) -> None:
        """문서 폴더(원본 PDF, 그림, 임베딩)를 지운다. 처리 중이면 거절."""
        import shutil
        with self._lock:
            st = self._status.get(doc_id)
            if st is None:
                raise KeyError(doc_id)
            if st["status"] == "processing":
                raise RuntimeError("processing")
            self._status.pop(doc_id)
            self._searchers.pop(doc_id, None)
            self._figs.pop(doc_id, None)
        shutil.rmtree(self.doc_dir(doc_id), ignore_errors=True)
        log.info("[%s] 삭제", doc_id)

    # ------------------------------------------------------------ 검색
    def searcher(self, doc_id: str):
        from pdfsearch.search.search import FigureSearcher
        with self._lock:
            s = self._searchers.get(doc_id)
        if s is None:
            s = FigureSearcher(self.doc_dir(doc_id), embedder=_LockedEmbedder(self.engines))
            with self._lock:
                self._searchers[doc_id] = s
        return s

    def figures(self, doc_id: str) -> dict[str, dict]:
        with self._lock:
            figs = self._figs.get(doc_id)
        if figs is None:
            recs = read_jsonl(self.doc_dir(doc_id) / "features" / "figures.jsonl")
            figs = {r["fig_id"]: r for r in recs}
            with self._lock:
                self._figs[doc_id] = figs
        return figs

    def search(self, doc_id: str, rgb: np.ndarray, k: int, mode: str) -> list[dict]:
        return self.searcher(doc_id).search(rgb, mode=mode, k=k)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class _IngestError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def default_config() -> Config:
    """tablet_backend/ 밖에서 실행해도 가중치를 찾도록 절대경로로."""
    cfg = Config()
    w = cfg.weights_path
    return replace(cfg, weights_path=w if w.is_absolute() else BASE_DIR / w)
