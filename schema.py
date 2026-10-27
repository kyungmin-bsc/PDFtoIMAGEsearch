"""모듈 간 데이터 계약.

좌표 규약 (불변):
  - 내부 좌표는 전부 xyxy = (x0, y0, x1, y1), 좌상단 원점, x 오른쪽 / y 아래 양수.
  - 기준 공간 = '원본 렌더 이미지' 픽셀 (PageItem.image, deskew 적용 후).
  - x1, y1 은 exclusive  ->  crop = image[y0:y1, x0:x1]
  - xyxy_norm = xyxy_px / (W, H)  ∈ [0, 1], DPI 독립.
  - xywh 는 사용하지 않는다.

페이지 규약:
  - pdf_index : 0-based PDF 내부 인덱스 (유일 키)
  - page_label: 책에 인쇄된 쪽 번호 (PDF에 정의 없으면 str(pdf_index + 1))
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class PageItem:
    pdf_index: int
    page_label: str
    image: np.ndarray            # H x W x 3, uint8, RGB  (crop 소스)
    dpi: int
    size_pt: tuple[float, float] # (W, H) in PDF point (1/72 inch), 회전 반영 후
    is_scanned: bool = False
    deskew_angle: float = 0.0    # [deg] image 에 이미 적용된 보정각

    @property
    def size_px(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return (w, h)


@dataclass
class Box:
    xyxy_px: tuple[int, int, int, int]
    cls: str
    score: float
    box_id: int = -1             # 읽기 순서 정렬 후 부여 (1-based)

    # ---------- 기하 유틸 ----------
    @property
    def w(self) -> int: return self.xyxy_px[2] - self.xyxy_px[0]
    @property
    def h(self) -> int: return self.xyxy_px[3] - self.xyxy_px[1]
    @property
    def area(self) -> int: return max(0, self.w) * max(0, self.h)

    def norm(self, size_px: tuple[int, int]) -> tuple[float, float, float, float]:
        W, H = size_px
        x0, y0, x1, y1 = self.xyxy_px
        return (round(x0 / W, 6), round(y0 / H, 6), round(x1 / W, 6), round(y1 / H, 6))

    def clip(self, size_px: tuple[int, int]) -> "Box":
        W, H = size_px
        x0, y0, x1, y1 = self.xyxy_px
        x0, y0 = max(0, min(x0, W)), max(0, min(y0, H))
        x1, y1 = max(0, min(x1, W)), max(0, min(y1, H))
        return Box((x0, y0, x1, y1), self.cls, self.score, self.box_id)


def iou(a: Box, b: Box) -> float:
    ix0, iy0 = max(a.xyxy_px[0], b.xyxy_px[0]), max(a.xyxy_px[1], b.xyxy_px[1])
    ix1, iy1 = min(a.xyxy_px[2], b.xyxy_px[2]), min(a.xyxy_px[3], b.xyxy_px[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def contain_ratio(small: Box, big: Box) -> float:
    """inter / area(small). 1.0 이면 small 이 big 안에 완전히 포함."""
    ix0, iy0 = max(small.xyxy_px[0], big.xyxy_px[0]), max(small.xyxy_px[1], big.xyxy_px[1])
    ix1, iy1 = min(small.xyxy_px[2], big.xyxy_px[2]), min(small.xyxy_px[3], big.xyxy_px[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return inter / small.area if small.area > 0 else 0.0


def union_box(a: Box, b: Box) -> Box:
    """두 박스를 감싸는 최소 박스. score 는 max (병합 결과의 대표 신뢰도)."""
    return Box(
        (min(a.xyxy_px[0], b.xyxy_px[0]), min(a.xyxy_px[1], b.xyxy_px[1]),
         max(a.xyxy_px[2], b.xyxy_px[2]), max(a.xyxy_px[3], b.xyxy_px[3])),
        a.cls, max(a.score, b.score),
    )
