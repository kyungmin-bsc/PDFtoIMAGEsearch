"""Box 정제. 적용 순서가 결과를 결정하므로 고정:
  1) conf 필터  2) NMS(클래스 무관)  3) 포함 흡수(큰 박스 유지)
  4) 조각 병합(축 겹침 + 근접)  5) 최소 면적  6) 패딩+클립  7) 읽기순서 + box_id
"""
from __future__ import annotations
import logging

from config import Config
from schema import Box, iou, contain_ratio, union_box

log = logging.getLogger(__name__)


def _nms(boxes: list[Box], th: float) -> list[Box]:
    keep: list[Box] = []
    for b in sorted(boxes, key=lambda x: -x.score):
        if all(iou(b, k) < th for k in keep):
            keep.append(b)
    return keep


def _absorb_contained(boxes: list[Box], th: float) -> list[Box]:
    """작은 박스가 큰 박스에 th 이상 포함 -> 큰 박스를 union으로 확장하고 작은 박스 제거."""
    boxes = sorted(boxes, key=lambda x: -x.area)
    out: list[Box] = []
    for b in boxes:
        for i, big in enumerate(out):
            if contain_ratio(b, big) >= th:
                out[i] = union_box(big, b)
                break
        else:
            out.append(b)
    return out


def _axis_overlap(a0, a1, b0, b1) -> float:
    """1D 겹침 길이 / 짧은 쪽 길이."""
    inter = max(0, min(a1, b1) - max(a0, b0))
    short = min(a1 - a0, b1 - b0)
    return inter / short if short > 0 else 0.0


def _should_merge(a: Box, b: Box, gap: int, min_axis: float = 0.5) -> bool:
    ax0, ay0, ax1, ay1 = a.xyxy_px
    bx0, by0, bx1, by1 = b.xyxy_px
    dx = max(0, max(ax0, bx0) - min(ax1, bx1))   # 가로 간격 (겹치면 0)
    dy = max(0, max(ay0, by0) - min(ay1, by1))   # 세로 간격
    # 좌우로 붙음: 세로 범위가 충분히 겹치고 가로 간격이 gap 이하
    if dy == 0 and dx <= gap and _axis_overlap(ay0, ay1, by0, by1) >= min_axis:
        return True
    # 상하로 붙음: 가로 범위가 충분히 겹치고 세로 간격이 gap 이하
    if dx == 0 and dy <= gap and _axis_overlap(ax0, ax1, bx0, bx1) >= min_axis:
        return True
    return False


def _merge_fragments(boxes: list[Box], gap: int) -> list[Box]:
    """고정점까지 반복 병합 (병합 결과가 새 이웃과 다시 붙을 수 있음)."""
    boxes = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _should_merge(boxes[i], boxes[j], gap):
                    boxes[i] = union_box(boxes[i], boxes[j])
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _pad(b: Box, p: int, size_px) -> Box:
    x0, y0, x1, y1 = b.xyxy_px
    return Box((x0 - p, y0 - p, x1 + p, y1 + p), b.cls, b.score).clip(size_px)


def _reading_order(boxes: list[Box], W: int) -> list[Box]:
    """단 구분: 페이지 폭 55% 미만 + 중심이 좌/우 절반 -> 0/1단, 넓은 박스는 단 구분 없이 y 순.
    키 = (y 밴드 내 위치) 대신 단순하고 예측 가능한 규칙:
      전폭 박스 -> col=-1 처리하지 않고 y0 기준으로 끼워 넣기 위해 (y0, col) 정렬은 부적절
      -> (col, y0) 정렬, 전폭 박스는 col=0 으로 간주."""
    def col(b: Box) -> int:
        cx = (b.xyxy_px[0] + b.xyxy_px[2]) / 2
        return 1 if (b.w < 0.55 * W and cx > W / 2) else 0
    return sorted(boxes, key=lambda b: (col(b), b.xyxy_px[1], b.xyxy_px[0]))


def postprocess(boxes: list[Box], size_px: tuple[int, int], cfg: Config) -> list[Box]:
    W, H = size_px
    short = min(W, H)
    n0 = len(boxes)

    b = [x for x in boxes if x.score >= cfg.conf_final]
    b = _nms(b, cfg.nms_iou)
    b = _absorb_contained(b, cfg.contain_ratio)
    b = _merge_fragments(b, gap=int(cfg.merge_gap_ratio * short))
    b = _absorb_contained(b, cfg.contain_ratio)          # 병합 후 새로 생긴 포함관계 정리
    b = [x for x in b if x.area >= cfg.min_area_ratio * W * H]
    b = [_pad(x, int(cfg.pad_ratio * short), size_px) for x in b]
    b = _reading_order(b, W)
    for i, x in enumerate(b, 1):
        x.box_id = i

    log.debug("postprocess: %d -> %d", n0, len(b))
    return b
