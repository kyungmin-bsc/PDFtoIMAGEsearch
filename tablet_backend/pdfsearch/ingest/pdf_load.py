"""PDF -> 페이지 단위 RGB 렌더 (generator).

책임:
  - 페이지별 고정 DPI 렌더 (회전 /Rotate 반영된 '보이는 그대로')
  - pdf_index(0-based) / page_label 추적
  - 스캔본 판정 + 원본 스캔 DPI 추정 -> 업샘플링 경고
책임 아님:
  - 출력 경로, 전처리(deskew), 추론
"""
from __future__ import annotations
import logging
from typing import Iterable, Iterator

import pymupdf as fitz
import numpy as np

from pdfsearch.common.config import Config
from pdfsearch.common.schema import PageItem

log = logging.getLogger(__name__)

SCAN_COVERAGE_TH = 0.85      # 단일 이미지가 페이지 면적의 85% 이상 덮으면 스캔본
MAX_PIXELS = 12_000 * 12_000 # 비정상적으로 큰 페이지(포스터 등) 방어


# ---------------------------------------------------------------- 유틸
def parse_page_range(spec: str | None, n_pages: int) -> list[int]:
    """'1-5,8,10-12' (1-based, 사람 기준) -> 0-based 인덱스 리스트. None이면 전체."""
    if not spec:
        return list(range(n_pages))
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b) if b else n_pages
        else:
            lo = hi = int(part)
        if lo < 1 or hi > n_pages or lo > hi:
            raise ValueError(f"페이지 범위 오류: '{part}' (문서 페이지 수 {n_pages})")
        out.update(range(lo - 1, hi))
    return sorted(out)


def _page_label(page: fitz.Page) -> str:
    try:
        lbl = page.get_label()
    except Exception:
        lbl = ""
    return lbl if lbl else str(page.number + 1)


def _scan_info(page: fitz.Page) -> tuple[bool, float | None]:
    """(is_scanned, native_dpi). 
    텍스트 유무로 판정하지 않는다: OCR된 스캔본은 투명 텍스트 레이어가 있기 때문.
    대신 '단일 래스터 이미지가 페이지 대부분을 덮는가'로 판정."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return False, None
    best_cov, best_dpi = 0.0, None
    for img in page.get_images(full=True):
        xref, w_px = img[0], img[2]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            r = r & page.rect                      # 페이지 밖으로 나간 부분 제거
            if r.is_empty:
                continue
            cov = (r.width * r.height) / page_area
            if cov > best_cov:
                best_cov = cov
                # 이미지 회전 배치 가능성 -> 긴 변끼리 비교
                long_pt = max(r.width, r.height)
                long_px = max(img[2], img[3])
                best_dpi = long_px / (long_pt / 72.0) if long_pt > 0 else None
    return best_cov >= SCAN_COVERAGE_TH, best_dpi


def _render(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    w, h = page.rect.width * zoom, page.rect.height * zoom
    if w * h > MAX_PIXELS:
        raise MemoryError(f"page {page.number}: {w:.0f}x{h:.0f}px 초과 (dpi 낮추세요)")
    # get_pixmap 은 /Rotate 를 자동 반영한다 (page.rect 도 회전 후 기준).
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                          colorspace=fitz.csRGB, alpha=False)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width * pix.n]  # stride 패딩 방어
    return arr.reshape(pix.height, pix.width, pix.n).copy()           # pix 해제 후에도 유효


# ---------------------------------------------------------------- 공개 API
def page_count(pdf_path: str) -> int:
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def iter_pages(pdf_path: str, cfg: Config,
               indices: Iterable[int] | None = None) -> Iterator[PageItem]:
    """페이지를 하나씩 yield. 한 번에 1페이지만 메모리에 상주."""
    doc = fitz.open(pdf_path)
    try:
        if doc.needs_pass:
            raise PermissionError(f"암호화된 PDF: {pdf_path}")
        idxs = list(indices) if indices is not None else range(doc.page_count)
        warned = False
        for i in idxs:
            page = doc.load_page(i)
            is_scanned, native_dpi = _scan_info(page)
            if (is_scanned and native_dpi and cfg.max_scan_dpi_warn
                    and native_dpi < cfg.dpi * 0.9 and not warned):
                log.warning("스캔 원본 DPI≈%.0f < 렌더 DPI %d: 업샘플링(정보 증가 없음, 용량만 증가)",
                            native_dpi, cfg.dpi)
                warned = True                       # 문서당 1회만
            yield PageItem(
                pdf_index=i,
                page_label=_page_label(page),
                image=_render(page, cfg.dpi),
                dpi=cfg.dpi,
                size_pt=(page.rect.width, page.rect.height),
                is_scanned=is_scanned,
            )
    finally:
        doc.close()
