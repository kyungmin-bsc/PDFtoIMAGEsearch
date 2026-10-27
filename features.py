"""그림/스케치 -> DINOv2 ViT-S/14 전역 임베딩 (384-d, L2 정규화).

normalize() 는 인덱싱과 검색이 공유하는 '단 하나의' 전처리. 수정 시 dataset 재생성 필요.
두 모드를 모두 저장, Step 9 평가 후 하나로 확정:
  raw : 회색조 -> 여백 trim -> 정사각 패딩 -> 224
  edge: 위 + Canny 선화(굵기 정규화) -> 스케치와 같은 형태로
출력:
  features/embeddings_raw.npy, embeddings_edge.npy  (N, 384) float16
  features/meta.jsonl   행 i <-> embeddings[i]  {"fig_id","file","page"}
  features/model.json   모델·전처리 정의
"""
from __future__ import annotations
import io, json, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from saver import imread_rgb, read_jsonl, atomic_write_text

MODEL_REPO, MODEL_NAME = "facebookresearch/dinov2", "dinov2_vits14"
INPUT = 224                          # 14의 배수
EDGE_WORK = 448                      # 엣지 추출 작업 해상도
TRIM_TH = 200                        # 이보다 어두운 픽셀 = 잉크
MARGIN = 0.05                        # 정사각 패딩 여백 비율
CANNY = (50, 150)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODES = ("raw", "edge")
PREPROC_VERSION = 1                  # normalize 바꾸면 +1


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
    if mode not in MODES:
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


def build_features(root: Path, emb: Embedder | None = None, batch: int = 16) -> int:
    root = Path(root)
    fd = root / "features"
    recs = {r["fig_id"]: r for r in read_jsonl(fd / "figures.jsonl")}   # 중복 시 마지막
    rows = sorted(recs.values(), key=lambda r: (r["page"], r["fig_id"]))
    rows = [r for r in rows if (root / r["file"]).exists()]
    emb = emb or Embedder()

    t = time.perf_counter()
    imgs = [imread_rgb(root / r["file"]) for r in rows]
    for mode in MODES:
        E = emb.embed_images(imgs, mode, batch).astype(np.float16)
        _save_npy(fd / f"embeddings_{mode}.npy", E)

    atomic_write_text(fd / "meta.jsonl", "".join(
        json.dumps({"fig_id": r["fig_id"], "file": r["file"], "page": r["page"],
                    "level": r.get("level", "figure"), "parent": r.get("parent")}) + "\n"
        for r in rows))

    atomic_write_text(fd / "model.json", json.dumps({
        "model": f"{MODEL_REPO}:{MODEL_NAME}", "dim": emb.dim, "dtype": "float16",
        "metric": "cosine (L2-normalized, dot product)",
        "modes": list(MODES), "preproc_version": PREPROC_VERSION,
        "preproc": {"input": INPUT, "edge_work": EDGE_WORK, "trim_th": TRIM_TH,
                    "margin": MARGIN, "canny": list(CANNY), "dilate": "3x3x1",
                    "mean": MEAN.tolist(), "std": STD.tolist()},
        "n": len(rows), "torch": torch.__version__,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"features: {len(rows)} figs x {len(MODES)} modes, {time.perf_counter()-t:.1f}s")
    return len(rows)
