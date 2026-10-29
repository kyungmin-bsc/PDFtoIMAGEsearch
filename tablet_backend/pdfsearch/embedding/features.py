"""그림/스케치 -> DINOv2 ViT-S/14 전역 임베딩 (384-d, L2 정규화).

normalize() 는 인덱싱과 검색이 공유하는 '단 하나의' 전처리. 수정 시 dataset 재생성 필요.
저장·검색 모드는 edge 하나만 쓴다 (MODES):
  edge: 회색조 -> 여백 trim -> 정사각 패딩 -> Canny 선화(굵기 정규화) -> 224, 스케치와 같은 형태
  raw : (edge 의 Canny 전 단계) normalize() 에만 남겨 둠, 임베딩은 만들지 않음
격자 조각 (index_version 2):
  손스케치는 그림 전체가 아니라 특징 부분만 그리는 경우가 많다.
  그래서 level=figure 그림은 전체 1장 + 여러 크기 격자 조각도 따로 임베딩한다.
    - 조각 한 변 = 잉크 영역 긴 변 x TILE_SCALES, TILE_OVERLAP 만큼 겹치게
    - 잉크(어두운 픽셀) 비율이 TILE_MIN_INK 미만인 조각(빈 공간)은 버림
  검색 시 그림 점수 = 그 그림의 (전체 + 조각) 중 최고 점수.
  조각은 normalize() 를 그대로 거치므로 PREPROC_VERSION 은 그대로다.
출력:
  features/embeddings_edge.npy  (N, 384) float16
  features/meta.jsonl   행 i <-> embeddings[i]
      {"fig_id","file","page","level","parent","kind": "whole"|"tile",
       "box": 조각 위치 [x, y, w, h] (그림 이미지 기준 0~1) | null}
  features/model.json   모델·전처리·조각 설정

이미 만든 문서에 조각 추가 (그림 다시 검출할 필요 없음):
  python -m pdfsearch.embedding.features server_data/documents/<document_id>
"""
from __future__ import annotations
import io, json, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pdfsearch.ingest.saver import imread_rgb, read_jsonl, atomic_write_text

MODEL_REPO, MODEL_NAME = "facebookresearch/dinov2", "dinov2_vits14"
INPUT = 224                          # 14의 배수
EDGE_WORK = 448                      # 엣지 추출 작업 해상도
TRIM_TH = 200                        # 이보다 어두운 픽셀 = 잉크
MARGIN = 0.05                        # 정사각 패딩 여백 비율
CANNY = (50, 150)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODES = ("edge",)                    # 저장·검색에 쓰는 모드 (raw 는 임베딩하지 않음)
PREPROC_VERSION = 1                  # normalize 바꾸면 +1

# ---- 격자 조각 ----
INDEX_VERSION = 2                    # 1: 그림당 1행 / 2: 그림 + 격자 조각
TILE_SCALES = (0.5, 0.33)            # 잉크 영역 긴 변 대비 조각 한 변
TILE_OVERLAP = 0.5                   # 이웃 조각과 겹치는 비율
TILE_MIN_INK = 0.02                  # 잉크 비율이 이보다 낮으면 빈 조각 -> 버림
TILE_MIN_PX = 48                     # 조각 짧은 변 최소 픽셀


# ---------------------------------------------------------------- 공유 전처리
def _trim_pad_square(gray: np.ndarray) -> np.ndarray:
    ink = gray < TRIM_TH
    if ink.any():
        ys, xs = np.where(ink)
        gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = gray.shape
    s = max(h, w)
    m = max(1, int(s * MARGIN))
    canvas = np.full((s + 2 * m, s + 2 * m), 255, np.uint8)
    oy, ox = m + (s - h) // 2, m + (s - w) // 2
    canvas[oy:oy + h, ox:ox + w] = gray
    return canvas


def normalize(rgb: np.ndarray, mode: str) -> np.ndarray:
    """RGB uint8 (임의 크기) -> 224x224 gray uint8, 흰 배경/검은 선."""
    if mode not in ("raw", "edge"):
        raise ValueError(mode)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    sq = _trim_pad_square(gray)
    if mode == "raw":
        return cv2.resize(sq, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    big = cv2.resize(sq, (EDGE_WORK, EDGE_WORK), interpolation=cv2.INTER_AREA)
    big = cv2.GaussianBlur(big, (3, 3), 0)
    e = cv2.Canny(big, *CANNY)
    e = cv2.dilate(e, np.ones((3, 3), np.uint8), iterations=1)   # 선 굵기 정규화
    return cv2.resize(255 - e, (INPUT, INPUT), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- 격자 조각
def _starts(lo: int, hi: int, size: int, stride: int) -> list[int]:
    """[lo, hi) 구간을 size 창으로 stride 씩. 마지막 창은 끝에 딱 맞춘다."""
    last = hi - size
    s = list(range(lo, last + 1, stride))
    if s[-1] != last:
        s.append(last)
    return s


def tile_boxes(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """그림 -> 조각 박스 xyxy (그림 이미지 픽셀). 빈 조각, 전체와 같은 조각은 제외."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    ink = gray < TRIM_TH
    if not ink.any():
        return []
    ys, xs = np.where(ink)
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    W, H = x1 - x0, y1 - y0
    L = max(W, H)

    out = set()
    for sc in TILE_SCALES:
        side = int(round(L * sc))
        tw, th = min(side, W), min(side, H)
        if min(tw, th) < TILE_MIN_PX or (tw == W and th == H):
            continue
        sx = max(1, int(tw * (1 - TILE_OVERLAP)))
        sy = max(1, int(th * (1 - TILE_OVERLAP)))
        for ty in _starts(y0, y1, th, sy):
            for tx in _starts(x0, x1, tw, sx):
                if ink[ty:ty + th, tx:tx + tw].mean() >= TILE_MIN_INK:
                    out.add((tx, ty, tx + tw, ty + th))
    return sorted(out, key=lambda b: (-(b[2] - b[0]), b[1], b[0]))   # 큰 조각부터


# ---------------------------------------------------------------- 임베더
class Embedder:
    def __init__(self, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        self.model = torch.hub.load(MODEL_REPO, MODEL_NAME, trust_repo=True).eval()
        self.dim = self.model.embed_dim            # 384

    @staticmethod
    def _to_tensor(grays: list[np.ndarray]) -> torch.Tensor:
        x = np.stack(grays).astype(np.float32) / 255.0              # (B,H,W)
        x = np.repeat(x[..., None], 3, axis=-1)                     # (B,H,W,3)
        x = (x - MEAN) / STD
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()

    @torch.inference_mode()
    def embed(self, grays: list[np.ndarray], batch: int = 16) -> np.ndarray:
        outs = []
        for i in range(0, len(grays), batch):
            z = self.model(self._to_tensor(grays[i:i + batch]))      # CLS (B,384)
            outs.append(F.normalize(z, dim=1).numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.dim), np.float32)

    def embed_images(self, rgbs: list[np.ndarray], mode: str, batch: int = 16) -> np.ndarray:
        return self.embed([normalize(r, mode) for r in rgbs], batch)


# ---------------------------------------------------------------- 인덱스 구축
def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def build_features(root: Path, emb: Embedder | None = None, batch: int = 16,
                   tiles: bool = True) -> int:
    """반환: 검색 대상 그림 수 (조각 행은 세지 않음)."""
    root = Path(root)
    fd = root / "features"
    recs = {r["fig_id"]: r for r in read_jsonl(fd / "figures.jsonl")}   # 중복 시 마지막
    rows = sorted(recs.values(), key=lambda r: (r["page"], r["fig_id"]))
    rows = [r for r in rows if (root / r["file"]).exists()]
    emb = emb or Embedder()

    t = time.perf_counter()
    crops, metas = [], []                          # 임베딩 행 i <-> (crops[i], metas[i])
    for r in rows:
        img = imread_rgb(root / r["file"])
        base = {"fig_id": r["fig_id"], "file": r["file"], "page": r["page"],
                "level": r.get("level", "figure"), "parent": r.get("parent")}
        crops.append(img)
        metas.append({**base, "kind": "whole", "box": None})
        if tiles and base["level"] == "figure":    # 세부 그림은 부모 조각이 이미 덮음
            h, w = img.shape[:2]
            for x0, y0, x1, y1 in tile_boxes(img):
                crops.append(np.ascontiguousarray(img[y0:y1, x0:x1]))
                metas.append({**base, "kind": "tile",
                              "box": [round(x0 / w, 4), round(y0 / h, 4),
                                      round((x1 - x0) / w, 4), round((y1 - y0) / h, 4)]})

    for mode in MODES:
        E = emb.embed_images(crops, mode, batch).astype(np.float16)
        _save_npy(fd / f"embeddings_{mode}.npy", E)
    (fd / "embeddings_raw.npy").unlink(missing_ok=True)   # 예전 방식(raw+edge)으로 만든 파일 정리

    atomic_write_text(fd / "meta.jsonl", "".join(json.dumps(m) + "\n" for m in metas))

    atomic_write_text(fd / "model.json", json.dumps({
        "model": f"{MODEL_REPO}:{MODEL_NAME}", "dim": emb.dim, "dtype": "float16",
        "metric": "cosine (L2-normalized, dot product)",
        "modes": list(MODES), "preproc_version": PREPROC_VERSION,
        "preproc": {"input": INPUT, "edge_work": EDGE_WORK, "trim_th": TRIM_TH,
                    "margin": MARGIN, "canny": list(CANNY), "dilate": "3x3x1",
                    "mean": MEAN.tolist(), "std": STD.tolist()},
        "index_version": INDEX_VERSION if tiles else 1,
        "tiles": {"enabled": tiles, "scales": list(TILE_SCALES), "overlap": TILE_OVERLAP,
                  "min_ink": TILE_MIN_INK, "min_px": TILE_MIN_PX},
        "n": len(rows), "n_rows": len(metas), "torch": torch.__version__,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    n_tiles = sum(m["kind"] == "tile" for m in metas)
    print(f"features: 그림 {len(rows)}개 + 조각 {n_tiles}개 = {len(metas)}행 x {len(MODES)} modes, "
          f"{time.perf_counter()-t:.1f}s")
    return len(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="figures.jsonl 기준으로 임베딩만 다시 만든다.")
    ap.add_argument("root", help="dataset 또는 server_data/documents/<document_id>")
    ap.add_argument("--no-tiles", action="store_true", help="조각 없이 그림당 1행 (예전 방식)")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    build_features(Path(a.root), batch=a.batch, tiles=not a.no_tiles)
