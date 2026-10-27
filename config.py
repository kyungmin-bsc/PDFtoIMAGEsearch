"""전역 설정. 모든 임계값은 여기서만 정의한다 (Step 7 튜닝 대상)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # ---- 렌더링 ----
    dpi: int = 150                     # 렌더/크롭 DPI (모든 출력 동일)
    max_scan_dpi_warn: bool = True     # 스캔본 원본 DPI < dpi 이면 경고

    # ---- 전처리 (추론용 사본 전용) ----
    deskew: bool = True
    deskew_max_angle: float = 5.0      # [deg] 이보다 큰 추정각은 오검출로 보고 무시

    # ---- 검출 ----
    weights_path: Path = Path("weights/doclayout_yolo_docstructbench_imgsz1024.pt")
    imgsz: int = 1024
    conf_infer: float = 0.10           # 추론 단계: recall 우선으로 낮게
    iou_infer: float = 0.50
    device: str = "cpu"
    target_classes: tuple[str, ...] = ("figure",)

    # ---- 후처리 ----
    conf_final: float = 0.70           # 최종 신뢰도 컷
    nms_iou: float = 0.50              # 클래스 무관 NMS
    contain_ratio: float = 0.90        # inter / area(small) >= 이면 '포함' -> 큰 박스로 흡수
    merge_gap_ratio: float = 0.01      # 조각 병합 허용 간격 (페이지 짧은 변 대비)
    min_area_ratio: float = 0.003      # 페이지 면적 대비 최소 박스 면적
    pad_ratio: float = 0.01            # 패딩 (페이지 짧은 변 대비)

    # ---- 출력 ----
    output_dir: Path = Path("dataset")
    debug_viz: bool = False

    # ---- 세부 그림 분할 ----
    split_subfigs: bool = True
    split_bg_delta: int = 40          # 배경색과 이만큼 다르면 전경
    split_border_ratio: float = 0.01  # 배경 추정용 테두리 두께
    split_border_uniform: float = 0.85# 테두리가 이 비율 이상 배경색이어야 분할 시도
    split_dilate_ratio: float = 0.004
    split_noise_ratio: float = 0.00005
    split_link_ratio: float = 0.012
    split_min_child: float = 0.06
    split_overlap_tol: float = 0.15
    split_max_children: int = 8
    split_max_depth: int = 3
    split_min_side_px: int = 48



    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}

