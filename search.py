"""사용자 스케치/이미지 -> 유사 그림 검색 (검색 담당).

조원 저장소(PDFtoIMAGEsearch)의 features.py / saver.py 를 그대로 import 해서
같은 전처리(normalize)와 같은 DINOv2 모델(Embedder)로 질의 벡터를 만든다.

사용 예 (터미널):
    python search.py my_sketch.png --mode edge --k 3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from features import Embedder, PREPROC_VERSION, MODES
from saver import imread_rgb

ROOT = Path("dataset")
FEATURE_DIR = ROOT / "features"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def find_pdf(root: Path = ROOT) -> Path | None:
    """현재 임베딩을 만든 PDF 경로 (run_meta.json 기준, 없으면 dataset 안 첫 PDF)."""
    run_meta = root / "features" / "run_meta.json"
    if run_meta.exists():
        name = json.loads(run_meta.read_text(encoding="utf-8"))["pdf"]["file"]
        if (root / name).exists():
            return root / name
    pdfs = sorted(root.glob("*.pdf"))
    return pdfs[0] if pdfs else None


class FigureSearcher:
    """embeddings_{mode}.npy 와 meta.jsonl 을 읽어 두고 질의마다 코사인 유사도 계산."""

    def __init__(self, root: Path = ROOT, embedder: Embedder | None = None):
        self.root = Path(root)
        fd = self.root / "features"

        # 1) 전처리 버전 확인 (README 필수 규칙 3)
        model_info = json.loads((fd / "model.json").read_text(encoding="utf-8"))
        if model_info["preproc_version"] != PREPROC_VERSION:
            raise RuntimeError(
                f"전처리 버전 불일치: dataset={model_info['preproc_version']}, "
                f"features.py={PREPROC_VERSION} -> dataset 을 다시 만들어야 함")

        # 2) 메타데이터 + 두 모드 임베딩 로드 (float16 -> float32)
        self.meta = _read_jsonl(fd / "meta.jsonl")
        self.emb = {}
        for mode in MODES:
            E = np.load(fd / f"embeddings_{mode}.npy").astype(np.float32)
            if len(E) != len(self.meta):
                raise RuntimeError(f"{mode}: 임베딩 {len(E)}개 != meta {len(self.meta)}줄")
            self.emb[mode] = E

        # 3) DINOv2 는 무거우므로 한 번만 로드
        self.embedder = embedder or Embedder()
        self.pdf_path = find_pdf(self.root)

    def embed_query(self, rgb: np.ndarray, mode: str) -> np.ndarray:
        return self.embedder.embed_images([rgb], mode)[0]          # (384,) L2 정규화

    def search(self, rgb: np.ndarray, mode: str = "edge", k: int = 3) -> list[dict]:
        """RGB uint8 이미지 -> 같은 페이지 중복 제거한 상위 k개 페이지."""
        if mode not in self.emb:
            raise ValueError(f"mode 는 {MODES} 중 하나")
        q = self.embed_query(rgb, mode)
        sims = self.emb[mode] @ q                                   # 코사인 유사도

        results, seen = [], set()
        for i in np.argsort(-sims):
            m = self.meta[i]
            if m["page"] in seen:                                   # 같은 페이지는 1번만
                continue
            seen.add(m["page"])
            results.append({
                "rank": len(results) + 1,
                "score": float(sims[i]),
                "page": m["page"],                                  # PDF 뷰어 기준 1부터
                "fig_id": m["fig_id"],
                "file": str(self.root / m["file"]),
                "level": m.get("level", "figure"),
                "parent": m.get("parent"),
            })
            if len(results) == k:
                break
        return results

    def search_file(self, path: str | Path, mode: str = "edge", k: int = 3) -> list[dict]:
        return self.search(imread_rgb(Path(path)), mode, k)

    def render_page(self, page: int, dpi: int = 150) -> bytes | None:
        """검색된 PDF 페이지를 PNG 바이트로 렌더링."""
        if self.pdf_path is None:
            return None
        import pymupdf as fitz
        with fitz.open(self.pdf_path) as doc:
            return doc.load_page(page - 1).get_pixmap(dpi=dpi).tobytes("png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--mode", default="edge", choices=MODES)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()

    s = FigureSearcher()
    print(f"PDF: {s.pdf_path}")
    for r in s.search_file(a.image, a.mode, a.k):
        print(f"{r['rank']}위 | p.{r['page']} | sim={r['score']:.4f} | {r['fig_id']} ({r['level']})")
