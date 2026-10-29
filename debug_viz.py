"""박스 오버레이 시각화. 여러 박스 세트를 색으로 구분 (예: raw vs post)."""
from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np

from schema import Box, PageItem

COLORS = [(255, 0, 0), (0, 160, 0), (0, 0, 255), (255, 128, 0)]  # RGB


def draw(item: PageItem, box_sets: dict[str, list[Box]],
         out_path: str | Path, max_side: int = 1600) -> None:
    img = item.image.copy()
    t = max(2, int(min(item.size_px) / 400))          # DPI 무관 선 두께
    for k, (name, boxes) in enumerate(box_sets.items()):
        col = COLORS[k % len(COLORS)]
        for b in boxes:
            x0, y0, x1, y1 = b.xyxy_px
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, t)
            label = f"{name}:{b.cls} {b.score:.2f}" + (f" #{b.box_id}" if b.box_id > 0 else "")
            cv2.putText(img, label, (x0 + 4, max(y0 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, t * 0.4, col, max(1, t // 2))
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
