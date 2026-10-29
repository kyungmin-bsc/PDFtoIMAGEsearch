"""python main.py book/circuit_test_book.pdf [--overwrite] [--pages 1-50] [--skip-features]
PDF -> figures/*.png -> features/{embeddings_*.npy, meta.jsonl, model.json}
"""
from __future__ import annotations
import argparse, logging, shutil, sys, time
from dataclasses import replace
from pathlib import Path

from config import Config
from pdf_load import iter_pages, page_count, parse_page_range
from preprocess import preprocess
from detector_yolo import FigureDetector
from postprocess import postprocess
from saver import Saver, load_done_pages
from features import build_features

log = logging.getLogger("main")


def setup_logging(root: Path) -> None:
    (root / "features").mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.FileHandler(root / "features" / "run.log", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    for h in (fh, sh):
        h.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[fh, sh], force=True)


def run(pdf: str, cfg: Config, pages: str | None, overwrite: bool, skip_features: bool) -> None:
    root = Path(cfg.output_dir)
    if overwrite:                                   # (a) 정책: 교재 1권만 유지, PDF는 보존
        for d in ("figures", "features"):
            shutil.rmtree(root / d, ignore_errors=True)
    setup_logging(root)

    n = page_count(pdf)
    todo = [i for i in parse_page_range(pages, n) if i not in load_done_pages(root)]
    log.info("pdf=%s pages=%d todo=%d", Path(pdf).name, n, len(todo))

    det = FigureDetector(cfg)
    t0, fails = time.perf_counter(), []
    with Saver(cfg, pdf, det.weights_hash) as sv:
        for k, p in enumerate(iter_pages(pdf, cfg, todo), 1):
            try:
                p = preprocess(p, cfg)
                boxes = postprocess(det.detect(p), p.size_px, cfg)
                sv.save_page(p, boxes)
                log.info("[%d/%d] page %d: %d figs", k, len(todo), p.pdf_index + 1, len(boxes))
            except Exception:
                log.exception("page %d 실패 -> 건너뜀 (다음 실행 때 재시도)", p.pdf_index + 1)
                fails.append(p.pdf_index + 1)
    log.info("추출 완료 %.1fs, 실패 페이지=%s", time.perf_counter() - t0, fails or "없음")

    if not skip_features:
        build_features(root)                        # 항상 전체 다시 계산 (수십 초)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="dataset")
    ap.add_argument("--pages", default=None, help="예: 1-50,60 (1부터 셈)")
    ap.add_argument("--overwrite", action="store_true", help="figures/features 삭제 후 새로 만듦")
    ap.add_argument("--skip-features", action="store_true")
    a = ap.parse_args()
    run(a.pdf, replace(Config(), output_dir=Path(a.out)), a.pages, a.overwrite, a.skip_features)
