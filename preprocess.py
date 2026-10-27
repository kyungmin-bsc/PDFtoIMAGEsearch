"""스캔 페이지 기울기 보정 (deskew).

방식: projection profile.
  글자 줄이 수평일 때 행 방향 투영(가로 합)이 '줄/여백'으로 가장 날카롭게 갈린다
  -> 행 합의 분산이 최대가 되는 각도 = 보정각.
  coarse(0.5°) -> fine(0.05°) 2단 탐색, 저해상도 이진 이미지에서 수행.

정책:
  - is_scanned 페이지만 적용 (디지털 PDF는 정의상 기울기 0)
  - |angle| < MIN_APPLY_DEG 이면 미적용 (보간 블러 방지)
  - 탐색 경계(±max)에 붙은 해, 또는 분산 개선이 미미하면 신뢰 불가 -> 미적용
  - 회전 시 캔버스 확장 (모서리 내용 손실 방지), 배경 흰색
"""
from __future__ import annotations
import logging
from dataclasses import replace

import cv2
import numpy as np

from config import Config
from schema import PageItem

log = logging.getLogger(__name__)

EST_LONG_SIDE = 1000      # 추정용 다운스케일 긴 변 [px]
MIN_APPLY_DEG = 0.10      # 이보다 작으면 회전 안 함
MIN_GAIN = 1.02           # score(best)/score(0) 최소 개선비


# ---------------------------------------------------------------- 추정
def _binarize_small(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    s = EST_LONG_SIDE / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    # 잉크=255, 배경=0
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return bw


def _score(bw: np.ndarray, angle: float) -> float:
    h, w = bw.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rot = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    prof = rot.sum(axis=1, dtype=np.float64)
    return float(np.var(prof))


def estimate_skew(rgb: np.ndarray, max_angle: float) -> tuple[float, bool]:
    """(angle_deg, reliable). angle 은 cv2 규약(양수 = 반시계 회전으로 보정)."""
    bw = _binarize_small(rgb)
    if bw.mean() < 0.5:                    # 거의 빈 페이지
        return 0.0, False

    s0 = _score(bw, 0.0)
    coarse = np.arange(-max_angle, max_angle + 1e-9, 0.5)
    sc = [_score(bw, a) for a in coarse]
    a_c = float(coarse[int(np.argmax(sc))])

    fine = np.arange(a_c - 0.5, a_c + 0.5 + 1e-9, 0.05)
    fine = fine[np.abs(fine) <= max_angle]
    sf = [_score(bw, a) for a in fine]
    k = int(np.argmax(sf))
    a_f, s_f = float(fine[k]), sf[k]

    at_edge = abs(a_f) >= max_angle - 0.05
    gain = s_f / s0 if s0 > 0 else np.inf
    reliable = (not at_edge) and (gain >= MIN_GAIN or abs(a_f) < MIN_APPLY_DEG)
    return round(a_f, 2), reliable


# ---------------------------------------------------------------- 적용
def rotate_expand(rgb: np.ndarray, angle: float) -> np.ndarray:
    """캔버스 확장 회전. 배경 흰색."""
    h, w = rgb.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(np.ceil(h * s + w * c)), int(np.ceil(h * c + w * s))
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgb, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def preprocess(item: PageItem, cfg: Config) -> PageItem:
    if not (cfg.deskew and item.is_scanned):
        return item
    angle, ok = estimate_skew(item.image, cfg.deskew_max_angle)
    if not ok:
        log.info("p%d: deskew 추정 불신뢰 (angle=%.2f) -> 미적용", item.pdf_index, angle)
        return item
    if abs(angle) < MIN_APPLY_DEG:
        return item
    log.info("p%d: deskew %.2f°", item.pdf_index, angle)
    return replace(item, image=rotate_expand(item.image, angle), deskew_angle=angle)
