"""Crop 저장 + 기록. 구조:
  dataset/figures/p0010_f01.png        큰 박스 (level=figure)
  dataset/figures/p0010_f01_s01.png    세부 그림 (level=subfigure, parent=p0010_f01)
  dataset/features/figures.jsonl       그림 1개 = 1줄 -> features.py 입력
  dataset/features/pages.jsonl         처리 완료 페이지 (이어서 실행용)
  dataset/features/run_meta.json       로컬 절대 경로 없음
"""
from __future__ import annotations
import json, os, platform, time
from pathlib import Path

import cv2
import numpy as np

from config import Config
from schema import Box, PageItem
from detector_yolo import file_sha256
from subfig_split import split_figure


def _imwrite_png(path: Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise IOError(f"PNG 인코딩 실패: {path}")
    tmp = path.with_suffix(".png.tmp")
    buf.tofile(str(tmp))                       # 한글 경로 안전
    os.replace(tmp, path)


def imread_rgb(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _versions() -> dict:
    v = {"python": platform.python_version()}
    for mod in ("pymupdf", "cv2", "numpy", "torch", "doclayout_yolo"):
        try:
            m = __import__(mod)
            v[mod] = str(getattr(m, "__version__", getattr(m, "VersionBind", "?")))
        except Exception:
            v[mod] = None
    return v


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class Saver:
    def __init__(self, cfg: Config, pdf_path: str, weights_hash: str):
        self.cfg = cfg
        self.root = Path(cfg.output_dir)
        self.fig_dir = self.root / "figures"
        self.feat_dir = self.root / "features"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.feat_dir.mkdir(parents=True, exist_ok=True)
        self._figs = open(self.feat_dir / "figures.jsonl", "a", encoding="utf-8")
        self._pages = open(self.feat_dir / "pages.jsonl", "a", encoding="utf-8")
        self.meta = {
            "pdf": {"file": Path(pdf_path).name,
                    "sha256": file_sha256(Path(pdf_path), 16)},
            "weights_hash": weights_hash,
            "config": {k: (v.replace("\\", "/") if isinstance(v, str) else v)
                       for k, v in cfg.to_dict().items()},
            "versions": _versions(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "finished": None, "n_pages": 0, "n_figures": 0, "n_subfigures": 0,
        }
        self._write_meta()

    # ------------------------------------------------------------ 저장 1건
    def _write(self, fig_id: str, img: np.ndarray, page_no: int, level: str,
               parent: str | None, xyxy, score: float, item: PageItem) -> dict:
        rel = f"figures/{fig_id}.png"
        _imwrite_png(self.root / rel, img)
        W, H = item.size_px
        rec = {"fig_id": fig_id, "file": rel, "page": page_no,
               "level": level, "parent": parent,
               "debug": {"pdf_index": item.pdf_index, "page_label": item.page_label,
                         "score": round(float(score), 4), "xyxy_px": [int(v) for v in xyxy],
                         "size_px": [W, H], "dpi": item.dpi,
                         "deskew_angle": item.deskew_angle}}
        self._figs.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ------------------------------------------------------------ 페이지 1장
    def save_page(self, item: PageItem, boxes: list[Box]) -> list[dict]:
        W, H = item.size_px
        page_no = item.pdf_index + 1                       # 뷰어 기준 1-based
        pad = max(1, int(min(W, H) * self.cfg.pad_ratio))
        recs, n_sub = [], 0

        for b in boxes:
            x0, y0, x1, y1 = b.xyxy_px
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, (item.pdf_index, b.xyxy_px)
            fig_id = f"p{page_no:04d}_f{b.box_id:02d}"
            crop = item.image[y0:y1, x0:x1]
            recs.append(self._write(fig_id, crop, page_no, "figure", None,
                                    b.xyxy_px, b.score, item))

            if self.cfg.split_subfigs:
                ch, cw = crop.shape[:2]
                for k, (sx0, sy0, sx1, sy1) in enumerate(split_figure(crop, self.cfg), 1):
                    cx0, cy0 = max(0, sx0 - pad), max(0, sy0 - pad)
                    cx1, cy1 = min(cw, sx1 + pad), min(ch, sy1 + pad)
                    sub_xyxy = (x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1)   # 페이지 좌표
                    recs.append(self._write(f"{fig_id}_s{k:02d}", crop[cy0:cy1, cx0:cx1],
                                            page_no, "subfigure", fig_id,
                                            sub_xyxy, b.score, item))
                    n_sub += 1

        self._figs.flush()
        # 페이지 완료 표시는 모든 저장이 끝난 뒤 (중간 실패 시 재처리 대상)
        self._pages.write(json.dumps({"pdf_index": item.pdf_index, "page": page_no,
                                      "n": len(boxes), "n_sub": n_sub}) + "\n")
        self._pages.flush()
        self.meta["n_pages"] += 1
        self.meta["n_figures"] += len(boxes)
        self.meta["n_subfigures"] += n_sub
        return recs

    # ------------------------------------------------------------ 종료
    def close(self) -> None:
        self.meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._write_meta()
        self._figs.close()
        self._pages.close()

    def _write_meta(self) -> None:
        atomic_write_text(self.feat_dir / "run_meta.json",
                          json.dumps(self.meta, ensure_ascii=False, indent=2))

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue                                       # 중단으로 잘린 줄
    return out


def load_done_pages(root: Path) -> set[int]:
    return {r["pdf_index"] for r in read_jsonl(Path(root) / "features" / "pages.jsonl")}
