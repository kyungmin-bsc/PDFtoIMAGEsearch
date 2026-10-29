"""실제 서버 + 실제 모델로 앱과 같은 흐름을 점검한다 (서버를 켠 상태에서 실행).

    python -m pdfsearch.tools.smoke_test --server http://localhost:8001 --pdf book/circuit_test_book.pdf

하는 일
  1) PDF 업로드 -> 처리 완료까지 시간 측정
  2) 서버가 뽑은 그림 몇 개로 질의를 만들어 검색
       - same  : 그림 원본 그대로 (정답 쪽이 1위여야 정상)
       - sketch: 윤곽선만 남기고 선을 굵게/흔들어 손그림처럼 만든 것
  3) 1위 결과의 bbox 를 PDF 페이지 위에 그려 위치가 맞는지 이미지로 저장
결과: server_data/smoke/ 아래 summary.json, bbox_*.png, query_*.png
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from pdfsearch.ingest.saver import imread_rgb, read_jsonl

BASE = Path(__file__).resolve().parents[2]          # tablet_backend/


def to_sketch(rgb: np.ndarray, seed: int) -> np.ndarray:
    """그림 -> 손그림 비슷한 흑백 선화 (윤곽선 + 굵기 변화 + 약간의 뒤틀림)."""
    rng = np.random.default_rng(seed)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    s = 512 / max(h, w)
    gray = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=int(rng.integers(1, 3)))
    # 살짝 원근 뒤틀기 (손으로 그린 비율 오차)
    hh, ww = edges.shape
    j = 0.04
    src = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    dst = (src + rng.uniform(-j, j, src.shape) * [ww, hh]).astype(np.float32)
    edges = cv2.warpPerspective(edges, cv2.getPerspectiveTransform(src, dst), (ww, hh))
    out = 255 - edges
    return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)


def png_bytes(rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return buf.tobytes()


def draw_bbox_on_page(pdf: Path, page: int, bbox: list[float], label: str, out: Path) -> None:
    import pymupdf as fitz
    with fitz.open(pdf) as doc:
        pix = doc.load_page(page - 1).get_pixmap(dpi=90)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    H, W = img.shape[:2]
    x, y, w, h = bbox
    p0, p1 = (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H))
    cv2.rectangle(img, p0, p1, (47, 64, 212), 3)
    cv2.putText(img, label, (p0[0], max(18, p0[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (47, 64, 212), 2)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    buf.tofile(str(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8001")
    ap.add_argument("--pdf", default=str(BASE / "book" / "circuit_test_book.pdf"))
    ap.add_argument("--data", default=str(BASE / "server_data"), help="서버의 PDFSEARCH_DATA 폴더")
    ap.add_argument("--n", type=int, default=5, help="질의로 쓸 그림 수")
    a = ap.parse_args()

    api = a.server.rstrip("/") + "/api/v1"
    pdf = Path(a.pdf)
    out = Path(a.data) / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    c = httpx.Client(timeout=600)

    print(f"[1] 서버 확인: {c.get(api + '/health').json()}")

    t0 = time.perf_counter()
    r = c.post(api + "/documents", files={"file": (pdf.name, pdf.read_bytes(), "application/pdf")})
    r.raise_for_status()
    doc = r.json()
    doc_id = doc["document_id"]
    print(f"[2] 업로드 {r.status_code} id={doc_id} ({'새로 처리' if r.status_code == 202 else '이미 처리됨'})")
    last = None
    while doc["status"] == "processing":
        time.sleep(1)
        doc = c.get(f"{api}/documents/{doc_id}").json()
        if doc != last:
            print(f"    ... {doc['status']} {time.perf_counter() - t0:.0f}s")
            last = doc
    t_ingest = time.perf_counter() - t0
    print(f"    -> {doc['status']} ({t_ingest:.1f}s) 쪽={doc['page_count']} 그림={doc['figure_count']}")
    if doc["status"] != "ready":
        print("    처리 실패:", doc.get("error"))
        return 1

    recs = [r for r in read_jsonl(Path(a.data) / "documents" / doc_id / "features" / "figures.jsonl")
            if r.get("level") == "figure"]
    random.Random(0).shuffle(recs)
    recs = sorted(recs[: a.n], key=lambda r: r["page"])
    print(f"[3] 질의 그림 {len(recs)}개: {[r['fig_id'] for r in recs]}")

    rows, times = [], []
    for i, rec in enumerate(recs):
        rgb = imread_rgb(Path(a.data) / "documents" / doc_id / rec["file"])
        for kind, q in (("same", rgb), ("sketch", to_sketch(rgb, i))):
            if kind == "sketch":
                cv2.imencode(".png", q)[1].tofile(str(out / f"query_{rec['fig_id']}.png"))
            t = time.perf_counter()
            r = c.post(f"{api}/documents/{doc_id}/search",
                       files={"sketch": ("q.png", png_bytes(q), "image/png")}, data={"top_k": "3"})
            dt = time.perf_counter() - t
            times.append(dt)
            if r.status_code != 200:
                print(f"    {rec['fig_id']} {kind}: 오류 {r.status_code} {r.text}")
                continue
            res = r.json()["results"]
            pages = [x["page"] for x in res]
            rank = pages.index(rec["page"]) + 1 if rec["page"] in pages else None
            rows.append({"fig_id": rec["fig_id"], "page": rec["page"], "kind": kind,
                         "rank": rank, "top": [(x["page"], x["figure_id"], x["score"]) for x in res],
                         "sec": round(dt, 2)})
            mark = {1: "1위", 2: "2위", 3: "3위"}.get(rank, "없음")
            print(f"    {rec['fig_id']:<14} {kind:<6} 정답 {mark:<3} | "
                  + "  ".join(f"p{p}:{s:.3f}" for p, _, s in rows[-1]["top"]) + f" | {dt:.2f}s")
            if kind == "same" and res:
                top = res[0]
                draw_bbox_on_page(pdf, top["page"], top["bbox"], top["figure_id"],
                                  out / f"bbox_{rec['fig_id']}.png")

    def acc(kind, k):
        rs = [x for x in rows if x["kind"] == kind]
        return f"{sum(1 for x in rs if x['rank'] and x['rank'] <= k)}/{len(rs)}"

    summary = {
        "document_id": doc_id, "pdf": pdf.name, "ingest_sec": round(t_ingest, 1),
        "page_count": doc["page_count"], "figure_count": doc["figure_count"],
        "search_sec_avg": round(sum(times) / len(times), 2) if times else None,
        "search_sec_first": round(times[0], 2) if times else None,
        "same_top1": acc("same", 1), "sketch_top1": acc("sketch", 1), "sketch_top3": acc("sketch", 3),
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] 원본 질의 1위 {summary['same_top1']} | 스케치 질의 1위 {summary['sketch_top1']}, "
          f"3위 안 {summary['sketch_top3']} | 검색 평균 {summary['search_sec_avg']}s "
          f"(첫 검색 {summary['search_sec_first']}s)")
    print(f"    결과 이미지: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
