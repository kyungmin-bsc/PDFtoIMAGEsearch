"""DocLayout-YOLO 추론 -> Box 리스트 (target_classes 만).

출력 좌표: PageItem.image 픽셀, xyxy, x1/y1 exclusive (schema 규약).
ultralytics 가 letterbox 역변환을 수행하므로 추가 매핑 불필요
 -> test_step3 에서 '좌표 범위 ⊂ 이미지' 로 확인.
"""
from __future__ import annotations
import hashlib
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from config import Config
from schema import Box, PageItem

log = logging.getLogger(__name__)


def file_sha256(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


class FigureDetector:
    def __init__(self, cfg: Config):
        from doclayout_yolo import YOLOv10   # 무거운 import 지연
        if not Path(cfg.weights_path).exists():
            raise FileNotFoundError(f"가중치 없음: {cfg.weights_path} (download_weights.py 실행)")
        self.cfg = cfg
        self.model = YOLOv10(str(cfg.weights_path))
        self.names: dict[int, str] = dict(self.model.names)

        # 클래스 이름 -> id (하드코딩 금지: 가중치마다 다를 수 있음)
        inv = {v: k for k, v in self.names.items()}
        missing = [c for c in cfg.target_classes if c not in inv]
        if missing:
            raise ValueError(f"가중치에 없는 클래스 {missing}. 사용 가능: {list(inv)}")
        self.target_ids = {inv[c] for c in cfg.target_classes}
        self.weights_hash = file_sha256(Path(cfg.weights_path))
        log.info("model loaded: classes=%s target_ids=%s hash=%s",
                 self.names, sorted(self.target_ids), self.weights_hash)

    def detect(self, item: PageItem) -> list[Box]:
        bgr = cv2.cvtColor(item.image, cv2.COLOR_RGB2BGR)
        res = self.model.predict(
            bgr,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf_infer,
            iou=self.cfg.iou_infer,
            device=self.cfg.device,
            verbose=False,
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()

        W, H = item.size_px
        out: list[Box] = []
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
            if c not in self.target_ids:
                continue
            b = Box((math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1)),
                    self.names[c], float(s)).clip((W, H))
            if b.w > 0 and b.h > 0:
                out.append(b)
        return out
