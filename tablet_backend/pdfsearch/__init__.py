"""PDF 교재 그림 검출 + 스케치 유사 그림 검색 (태블릿 앱 백엔드용).

  common     설정(Config)·데이터 계약(Box, PageItem)
  ingest     PDF -> 그림 검출·크롭·저장   (python -m pdfsearch.ingest.main book/x.pdf)
  embedding  그림/스케치 -> DINOv2 임베딩
  search     스케치 -> 유사 그림 top-k    (FigureSearcher)
  sync       dataset <-> Supabase Storage (python -m pdfsearch.sync.supabase_sync)
  tools      개발·디버그용 (Streamlit 데모, 시각화)

모든 경로(weights/, dataset/, book/)는 tablet_backend/ 폴더 기준 상대경로이므로
tablet_backend/ 에서 실행할 것.
"""
