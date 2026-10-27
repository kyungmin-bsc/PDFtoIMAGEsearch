"""python check_same.py <질의 이미지> [--mode raw|edge] [--k 3]
dataset/features 에 저장된 벡터와 비교해서 상위 k개 '페이지'를 출력 (같은 페이지는 1번만).
"""
import argparse, json
from pathlib import Path
import numpy as np
from saver import imread_rgb
from features import Embedder

ap = argparse.ArgumentParser()
ap.add_argument("image")
ap.add_argument("--root", default="dataset")
ap.add_argument("--mode", default="raw", choices=["raw", "edge"])
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()

fd = Path(a.root) / "features"
E = np.load(fd / f"embeddings_{a.mode}.npy").astype(np.float32)
meta = [json.loads(l) for l in (fd / "meta.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(meta) == E.shape[0], "meta.jsonl 과 embeddings 행 수 불일치"

q = Embedder().embed_images([imread_rgb(Path(a.image))], a.mode)[0]
sims = E @ q

seen, rank = set(), 0
for i in np.argsort(-sims):
    m = meta[i]
    if m["page"] in seen:
        continue                       # 같은 페이지는 한 번만
    seen.add(m["page"]); rank += 1
    print(f"#{rank}  page {m['page']:>4}  sim={sims[i]:.4f}  ({m['fig_id']})")
    if rank == a.k:
        break
