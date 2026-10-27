"""큰 figure crop -> 세부 그림 박스 (배경 기반 재귀 분할). 분할 불가면 [].
반환: crop 기준 xyxy (x1/y1 exclusive).

핵심 아이디어: '잉크'가 아니라 '배경과 다른 픽셀'을 전경으로 본다.
  - 배경색 = crop 테두리 픽셀의 중앙값 (흰색/검정/색 배경 모두 대응)
  - 테두리가 균일하지 않으면(사진·그림이 가장자리까지 꽉 참) -> 원자 단위, 분할 안 함
  - 전경 연결 성분 -> 근접 그룹 -> 큰 그룹 = 세부 그림, 작은 그룹(라벨) 흡수
  - 각 세부 그림에 같은 과정을 재귀 (예: 검은 배경 펼침면 -> 흰 페이지 -> 개별 작품)
"""
from __future__ import annotations
import cv2
import numpy as np


# ---------------------------------------------------------------- 기하
def _area(b): return (b[2] - b[0]) * (b[3] - b[1])
def _union(a, b): return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
def _inter(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
def _gap(a, b):
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def _group(comps, link):
    parent = list(range(len(comps)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if _gap(comps[i], comps[j]) <= link:
                parent[find(i)] = find(j)
    g = {}
    for i, c in enumerate(comps):
        r = find(i)
        g[r] = _union(g[r], c) if r in g else c
    return list(g.values())


def _merge_overlaps(boxes, tol):
    boxes = list(boxes); changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if _inter(a, b) > tol * min(_area(a), _area(b)):
                    boxes[i] = _union(a, b); del boxes[j]; changed = True; break
            if changed:
                break
    return boxes


def _reading_order(boxes):
    rows = []
    for b in sorted(boxes, key=lambda b: b[1]):
        for r in rows:
            ref = r[0]
            ov = min(b[3], ref[3]) - max(b[1], ref[1])
            if ov >= 0.5 * min(b[3] - b[1], ref[3] - ref[1]):
                r.append(b); break
        else:
            rows.append([b])
    return [b for r in rows for b in sorted(r, key=lambda b: b[0])]


# ---------------------------------------------------------------- 배경 / 전경
def _background(img: np.ndarray, cfg):
    """(bg_color, uniform). 테두리 strip 픽셀의 중앙값과 그 색에 가까운 비율."""
    h, w = img.shape[:2]
    s = max(1, int(min(h, w) * cfg.split_border_ratio))
    border = np.concatenate([img[:s].reshape(-1, 3), img[-s:].reshape(-1, 3),
                             img[:, :s].reshape(-1, 3), img[:, -s:].reshape(-1, 3)]).astype(np.int16)
    bg = np.median(border, axis=0)
    near = (np.abs(border - bg).max(axis=1) <= cfg.split_bg_delta).mean()
    return bg, float(near)


def _foreground(img: np.ndarray, bg, cfg) -> np.ndarray:
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    fg = (diff > cfg.split_bg_delta).astype(np.uint8)
    fg = cv2.medianBlur(fg * 255, 3) // 255                     # 스캔 점 노이즈
    k = max(1, int(min(img.shape[:2]) * cfg.split_dilate_ratio))
    return cv2.dilate(fg, np.ones((k, k), np.uint8))


# ---------------------------------------------------------------- 한 단계 분할
def _split_once(img: np.ndarray, cfg) -> list[tuple]:
    h, w = img.shape[:2]
    if min(h, w) < cfg.split_min_side_px:
        return []
    bg, uniform = _background(img, cfg)
    if uniform < cfg.split_border_uniform:       # 가장자리까지 내용이 꽉 참 = 사진/그림 1장
        return []
    fg = _foreground(img, bg, cfg)
    _, _, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    comps = [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh, a in st[1:]
             if a >= cfg.split_noise_ratio * w * h]
    if len(comps) < 2:
        return []

    groups = _group(comps, cfg.split_link_ratio * min(h, w))
    min_a = cfg.split_min_child * w * h
    big = _merge_overlaps([g for g in groups if _area(g) >= min_a], cfg.split_overlap_tol)
    if not (2 <= len(big) <= cfg.split_max_children):
        return []
    for s in sorted((g for g in groups if _area(g) < min_a), key=_area, reverse=True):
        k = min(range(len(big)), key=lambda i: _gap(s, big[i]))
        big[k] = _union(big[k], s)
    big = _merge_overlaps(big, cfg.split_overlap_tol)
    return big if len(big) >= 2 else []


def _split_rec(img, cfg, depth, ox, oy, out, inset=0, parent_bg=None):
    """재귀 분할.
    inset    : 분석 시에만 가장자리 m px 무시 (경계 번짐 제거). 결과 박스는 원래 영역 기준.
    parent_bg: 바깥 단계의 배경색. 자식 배경이 이와 같으면 '같은 레벨'로 보고 더 쪼개지 않음
               (회로: 흰->흰 = 멈춤 / 펼침면: 검정->흰 = 계속).
    """
    h, w = img.shape[:2]
    parts, bg = [], None
    if depth > 0 and min(h, w) > 2 * inset:
        a = img[inset:h - inset, inset:w - inset] if inset else img
        bg, _ = _background(a, cfg)
        same_level = (parent_bg is not None and
                      np.abs(bg - parent_bg).max() <= cfg.split_bg_delta)
        if not same_level:
            parts = [(x0 + inset, y0 + inset, x1 + inset, y1 + inset)
                     for x0, y0, x1, y1 in _split_once(a, cfg)]
    if not parts:
        out.append((ox, oy, ox + w, oy + h))
        return
    m = max(2, int(min(h, w) * cfg.split_dilate_ratio) + 2)
    for x0, y0, x1, y1 in parts:
        _split_rec(img[y0:y1, x0:x1], cfg, depth - 1, ox + x0, oy + y0, out, m, bg)


def split_figure(rgb: np.ndarray, cfg) -> list[tuple[int, int, int, int]]:
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    leaves: list = []
    _split_rec(rgb, cfg, cfg.split_max_depth, 0, 0, leaves)
    if len(leaves) < 2:
        return []
    return [tuple(int(v) for v in b) for b in _reading_order(leaves)]
