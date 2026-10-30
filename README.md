# PDFtoIMAGEsearch

**손으로 그린 스케치로 PDF 교재 속 그림을 찾아 주는 검색 시스템**

PDF에서 그림(figure)을 자동으로 검출·추출해 임베딩해 두고, 사용자가 태블릿에 대충 그린 스케치와 가장 비슷한 그림이 있는 페이지를 찾아 줍니다.

```
PDF ──► 페이지 렌더링 ──► DocLayout-YOLO 그림 검출 ──► 후처리 / 세부 그림 분할 ──► figures/*.png
                                                                              │
스케치 ──► 선화 정규화(Canny) ──► DINOv2 임베딩 ──► 코사인 유사도 ◄── DINOv2 임베딩(그림 + 격자 조각)
                                                      │
                                               상위 k개 페이지 + 그림 위치(bbox)
```

---

## 주요 기능

- **그림 검출**: [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)(DocStructBench 가중치)로 페이지마다 `figure` 영역을 찾습니다. 낮은 신뢰도로 넓게 검출한 다음 후처리로 정제합니다(NMS → 포함 박스 흡수 → 조각 병합 → 최소 면적 → 패딩).
- **세부 그림 분할**: 한 박스 안에 그림이 여러 개 있으면 배경색을 기준으로 재귀 분할해 `subfigure`로 따로 저장합니다.
- **스케치 친화 임베딩**: 그림과 스케치를 같은 방식으로 선화(Canny edge)로 바꾼 뒤 DINOv2 ViT-S/14로 384차원 벡터를 만듭니다.
- **부분 스케치 검색**: 그림 전체 외에도 여러 크기의 **격자 조각(tile)**을 함께 임베딩합니다. 그래서 그림의 특징적인 부분만 그려도 찾을 수 있습니다.
- **태블릿 앱용 HTTP 서버**: FastAPI 서버에 PDF를 올리면 백그라운드에서 처리하고, 스케치 PNG를 보내면 결과를 돌려줍니다. 같은 와이파이에 있는 앱이 서버를 **UDP 브로드캐스트로 자동 검색**할 수 있습니다.
- **Supabase 동기화**: 만든 데이터셋(임베딩, 메타데이터)을 Supabase Storage 비공개 버킷에 올리고 내려받을 수 있습니다.

## 폴더 구조

현재 개발은 `tablet_backend/`의 `pdfsearch` 패키지에서 합니다. 루트에 있는 `*.py`는 패키지로 정리하기 전의 초기 버전입니다.

```
tablet_backend/
├── pdfsearch/
│   ├── common/      config.py (모든 임계값), schema.py (Box, PageItem)
│   ├── ingest/      PDF → 그림 추출
│   │                pdf_load · preprocess(deskew) · detector_yolo · postprocess · subfig_split · saver · main
│   ├── embedding/   features.py — 선화 정규화 + DINOv2 임베딩 + 격자 조각
│   ├── search/      search.py — FigureSearcher (스케치 → 유사 그림)
│   ├── server/      api.py (FastAPI) · store.py (문서 관리, 백그라운드 처리) · discovery.py (UDP 자동 검색)
│   ├── sync/        supabase_sync.py
│   └── tools/       search_app.py (Streamlit 데모) · smoke_test.py · debug_viz.py · evaluate.py
├── tests/           test_server.py (모델 없이 API 흐름 테스트)
├── book/            테스트용 PDF
├── dataset/         CLI로 만든 결과 (figures/, features/)
├── weights/         doclayout_yolo_docstructbench_imgsz1024.pt
└── requirements.txt
```

## 설치

Python 3.10 기준입니다.

```bash
cd tablet_backend
python -m venv .venv
.venv\Scripts\activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

- 검출 가중치(`weights/doclayout_yolo_docstructbench_imgsz1024.pt`)는 저장소에 들어 있습니다.
- DINOv2 모델은 첫 실행 때 `torch.hub`로 자동 다운로드됩니다(인터넷 필요).
- 기본값은 CPU 실행입니다(`Config.device = "cpu"`).

아래 명령은 모두 **`tablet_backend/` 폴더에서** 실행합니다.

## 사용법

### 1. PDF → 그림 추출 + 임베딩 (CLI)

```bash
python -m pdfsearch.ingest.main book/circuit_test_book.pdf
python -m pdfsearch.ingest.main book/circuit_test_book.pdf --pages 1-50,60   # 일부 페이지만
python -m pdfsearch.ingest.main book/circuit_test_book.pdf --overwrite       # 기존 결과 지우고 새로
```

| 옵션 | 설명 |
|---|---|
| `-o, --out` | 출력 폴더 (기본 `dataset`) |
| `--pages` | 처리할 페이지, 1부터 셈 (예: `1-50,60`) |
| `--overwrite` | `figures/`, `features/`를 지우고 다시 만듦 |
| `--skip-features` | 그림 추출만 하고 임베딩은 건너뜀 |

이미 처리한 페이지는 `pages.jsonl`에 기록돼 있어서, 중간에 멈춘 뒤 다시 실행하면 남은 페이지부터 이어서 처리합니다. 그림을 다시 검출하지 않고 임베딩만 새로 만들려면 아래처럼 실행합니다.

```bash
python -m pdfsearch.embedding.features dataset            # --no-tiles: 조각 없이 그림당 1행
```

### 2. 검색

```bash
# 터미널
python -m pdfsearch.search.search my_sketch.png --k 3

# Streamlit 데모 화면
python -m streamlit run pdfsearch/tools/search_app.py
```

결과는 같은 페이지를 한 번만 넣은 상위 k개 페이지입니다. 그림 전체와 일치했는지(`whole`), 일부 조각과 일치했는지(`tile`)와 함께 조각 위치도 알려 줍니다.

### 3. 태블릿 앱용 서버

```bash
uvicorn pdfsearch.server.api:app --host 0.0.0.0 --port 8001
```

서버를 켜면 앱에서 직접 입력할 수 있는 내부 IP 주소가 로그에 출력됩니다.

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/api/v1/documents` | PDF 업로드 (`file`, 최대 50MB) → `202 processing`. 같은 PDF면 `200` |
| `GET` | `/api/v1/documents` | 등록된 문서 목록 (용량, 등록 시각 포함) |
| `GET` | `/api/v1/documents/{document_id}` | 처리 상태 (`processing` / `ready` / `failed`) |
| `DELETE` | `/api/v1/documents/{document_id}` | 문서 삭제 (처리 중이면 거절) |
| `POST` | `/api/v1/documents/{document_id}/search` | 스케치 이미지(`sketch`) + `top_k`(1~10, 기본 3) → 유사 그림 목록 |
| `GET` | `/api/v1/documents/{document_id}/figures/{figure_id}/image?size=thumb\|full` | 그림 이미지 |
| `GET` | `/api/v1/health` | 서버 상태 |

검색 결과 예시:

```json
{
  "document_id": "3f2a9c...",
  "results": [
    {
      "figure_id": "p0010_f01",
      "score": 0.8123,
      "page": 10,
      "bbox": [0.12, 0.30, 0.55, 0.28],
      "image_url": "/api/v1/documents/3f2a9c.../figures/p0010_f01/image?size=thumb",
      "caption": null,
      "level": "figure"
    }
  ]
}
```

- `bbox`는 페이지 기준 `[x, y, w, h]`이고, 0~1 사이 값이며 원점은 좌상단입니다.
- 에러는 `{"error": {"code": "...", "message": "..."}}` 형식입니다. 코드: `DOCUMENT_NOT_FOUND`, `DOCUMENT_NOT_READY`, `NO_FIGURES`, `INVALID_FILE`, `EMPTY_SKETCH`, `FILE_TOO_LARGE`, `INTERNAL_ERROR`
- `document_id`는 PDF 내용의 SHA-256 앞 16자리입니다. 같은 PDF를 다시 올리면 처리하지 않고 바로 `ready`를 돌려줍니다.
- 문서 데이터는 `server_data/documents/<document_id>/`에 저장됩니다(`.gitignore` 대상).

**자동 서버 검색**: 앱이 UDP `8002`번으로 `YOONS_DISCOVER`를 브로드캐스트하면, 서버가 `{"service": "yoonsimage", "port": 8001}`로 응답합니다.

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `PDFSEARCH_DATA` | `tablet_backend/server_data` | 문서 저장 폴더 |
| `PDFSEARCH_PORT` | `8001` | 앱에 알려 줄 HTTP 포트 (`--port`와 같게) |
| `PDFSEARCH_DISCOVERY_PORT` | `8002` | 자동 검색용 UDP 포트 |
| `PDFSEARCH_MODE` | `edge` | 검색 모드 (임베딩을 edge만 저장하므로 바꾸지 않음) |

### 4. Supabase 동기화 (선택)

```bash
python -m pdfsearch.sync.supabase_sync upload   --root dataset --book book/book.pdf --prefix book1
python -m pdfsearch.sync.supabase_sync download --root dataset_cache --prefix book1 --search-only
```

`.env` 파일(깃에 올리지 않음)에 다음 값을 넣습니다.

```
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_KEY=...      # service_role 키 — 서버/로컬에서만 사용
SUPABASE_BUCKET=dataset
```

책 PDF 원본은 클라우드에 올리지 않습니다. 대신 manifest에 SHA-256과 쪽수를 기록해 두고, 로컬에 있는 PDF가 데이터셋을 만든 원본과 같은지 확인하는 데 씁니다.

## 출력 데이터 형식

```
dataset/
├── figures/
│   ├── p0010_f01.png          그림 (level=figure)
│   └── p0010_f01_s01.png      세부 그림 (level=subfigure, parent=p0010_f01)
└── features/
    ├── figures.jsonl          그림 1개 = 1줄 (페이지, 박스 좌표, level, parent)
    ├── pages.jsonl            처리 완료 페이지 (이어서 실행용)
    ├── embeddings_edge.npy    (N, 384) float16, L2 정규화
    ├── meta.jsonl             i번째 줄 ↔ embeddings[i]  {fig_id, page, level, kind: whole|tile, box}
    ├── model.json             모델, 전처리, 조각 설정, preproc_version
    ├── run_meta.json          원본 PDF 정보, 실행 설정
    └── run.log
```

> ⚠️ `features.py`의 `normalize()`는 인덱싱과 검색에 **똑같이** 쓰이는 전처리입니다. 이 함수를 바꾸면 `PREPROC_VERSION`을 올리고 데이터셋을 다시 만들어야 합니다. 버전이 맞지 않으면 검색기가 실행을 거부합니다.

## 설정 튜닝

검출, 후처리, 세부 그림 분할에 쓰는 임계값은 모두 `pdfsearch/common/config.py`의 `Config`에 모여 있습니다. 주요 항목:

| 항목 | 기본값 | 의미 |
|---|---|---|
| `dpi` | 150 | 렌더링·크롭 해상도 |
| `conf_infer` / `conf_final` | 0.10 / 0.70 | 검출 단계 / 최종 신뢰도 기준 |
| `nms_iou` | 0.50 | NMS IoU 기준 |
| `contain_ratio` | 0.90 | 이 비율 이상 포함된 박스는 큰 박스로 흡수 |
| `min_area_ratio` | 0.003 | 페이지 면적 대비 최소 그림 크기 |
| `split_subfigs` | True | 세부 그림 분할 사용 여부 |

격자 조각 설정(`TILE_SCALES`, `TILE_OVERLAP`, `TILE_MIN_INK`)은 `embedding/features.py` 맨 위에 있습니다.

## 테스트

```bash
# 모델 없이 API 흐름 테스트 (가짜 검출기·임베더 사용)
python -m pytest tests -q

# 실제 모델로 처음부터 끝까지 점검 (서버를 켠 상태에서)
python -m pdfsearch.tools.smoke_test --server http://localhost:8001 --pdf book/circuit_test_book.pdf
```

`smoke_test`는 추출한 그림을 손그림처럼 바꿔 검색해 보고, 1위 결과의 bbox를 PDF 페이지 위에 그린 이미지를 `server_data/smoke/`에 저장합니다.

## 사용 기술

PyMuPDF · OpenCV · DocLayout-YOLO · PyTorch / DINOv2 · FastAPI / Uvicorn · Streamlit · Supabase
