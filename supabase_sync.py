"""dataset/ <-> Supabase Storage (비공개 버킷) 동기화.

책 PDF는 로컬 book/ 폴더에만 두고 클라우드에는 올리지 않는다.
업로드할 때 PDF의 SHA-256과 쪽수를 manifest에 기록하고,
검색 쪽에서는 로컬 PDF가 데이터셋을 만든 원본과 같은지 이 값으로 확인한다.

사용:
  python supabase_sync.py upload   [--root dataset] [--book book/book.pdf] [--prefix book1]
  python supabase_sync.py download [--root dataset_cache] [--prefix book1] [--search-only]

환경변수 (.env 또는 셸):
  SUPABASE_URL=https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY=...   # service_role 키. 서버/로컬 PC에서만 사용. 앱이나 깃에 절대 넣지 말 것
  SUPABASE_BUCKET=dataset    # 기본값 dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import sys
from pathlib import Path

from supabase import create_client

FREE_PLAN_FILE_LIMIT = 50 * 1024 * 1024          # Free 플랜 단일 파일 업로드 한도 50MB
EXCLUDE_SUFFIXES = {".pdf"}                       # 책 원본은 로컬 book/ 에만 둠
SEARCH_FILES = (                                  # 검색 담당이 실제로 쓰는 파일
    "features/embeddings_raw.npy",
    "features/embeddings_edge.npy",
    "features/meta.jsonl",
    "features/model.json",
)
MANIFEST = "manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def client_and_bucket():
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_SERVICE_KEY 환경변수를 설정하세요.")
    bucket = os.environ.get("SUPABASE_BUCKET", "dataset")
    sb = create_client(url, key)
    names = {b.name for b in sb.storage.list_buckets()}
    if bucket not in names:
        sb.storage.create_bucket(bucket, options={"public": False})   # 비공개 버킷
        print(f"[bucket] '{bucket}' 생성 (private)")
    return sb, bucket


def remote_manifest(store, prefix: str) -> dict:
    """원격 manifest. 파일이 없을 때만 빈 manifest를 반환하고, 그 외 오류는 그대로 발생시킨다."""
    try:
        raw = store.download(f"{prefix}/{MANIFEST}")
    except Exception as e:
        msg = str(e).lower()
        if "not found" in msg or "not_found" in msg or "404" in msg:
            return {"files": {}}
        raise                                      # 네트워크/인증 오류는 숨기지 않음
    return json.loads(raw)


def book_info(book: Path) -> dict:
    import pymupdf as fitz
    with fitz.open(book) as doc:
        n_pages = doc.page_count
    return {"name": book.name, "sha256": sha256(book), "pages": n_pages}


def max_meta_page(root: Path) -> int:
    meta = root / "features" / "meta.jsonl"
    if not meta.exists():
        return 0
    return max((json.loads(l)["page"] for l in meta.read_text("utf-8").splitlines() if l.strip()),
               default=0)


def upload(root: Path, prefix: str, book: Path) -> None:
    if not root.is_dir():
        sys.exit(f"폴더 없음: {root}")
    if not book.is_file():
        sys.exit(f"책 PDF 없음: {book}")

    info = book_info(book)
    mp = max_meta_page(root)
    if mp > info["pages"]:
        sys.exit(f"meta.jsonl의 최대 page {mp}가 PDF 쪽수 {info['pages']}보다 큽니다: 다른 책입니다.")

    files = sorted(p for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() not in EXCLUDE_SUFFIXES)
    skipped = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXCLUDE_SUFFIXES]
    for p in skipped:
        print(f"[skip] 업로드 제외: {p}")

    too_big = [p for p in files if p.stat().st_size > FREE_PLAN_FILE_LIMIT]
    if too_big:
        for p in too_big:
            print(f"[경고] 50MB 초과 → Free 플랜 업로드 불가: {p} ({p.stat().st_size/2**20:.1f} MB)")
        sys.exit("큰 파일을 분할/압축하거나 플랜 한도를 확인하세요.")

    sb, bucket = client_and_bucket()
    store = sb.storage.from_(bucket)
    old = remote_manifest(store, prefix)["files"]

    total = sum(p.stat().st_size for p in files)
    print(f"[upload] {len(files)}개, {total/2**20:.1f} MB → {bucket}/{prefix}/")

    new, sent = {}, 0
    for p in files:
        rel = p.relative_to(root).as_posix()
        digest = sha256(p)
        new[rel] = {"sha256": digest, "size": p.stat().st_size}
        if old.get(rel, {}).get("sha256") == digest:
            continue                               # 변경 없음 → 건너뜀
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        store.upload(f"{prefix}/{rel}", p.read_bytes(),
                     file_options={"content-type": ctype, "upsert": "true"})
        sent += 1
        print(f"  ↑ {rel}")

    # 로컬에서 사라진 파일은 원격에서도 삭제 (교재 교체 시 섞임 방지)
    stale = [f"{prefix}/{r}" for r in old if r not in new]
    for i in range(0, len(stale), 100):
        store.remove(stale[i:i + 100])
    if stale:
        print(f"  ✗ 원격 잔여 파일 {len(stale)}개 삭제")

    model = root / "features" / "model.json"
    manifest = {
        "preproc_version": json.loads(model.read_text("utf-8")).get("preproc_version") if model.exists() else None,
        "book": info,
        "files": new,
    }
    store.upload(f"{prefix}/{MANIFEST}", json.dumps(manifest, ensure_ascii=False, indent=1).encode(),
                 file_options={"content-type": "application/json", "upsert": "true"})
    print(f"[done] 업로드 {sent}개, 변경 없음 {len(new)-sent}개, manifest 갱신 "
          f"(book={info['name']}, {info['pages']}쪽)")


def download(root: Path, prefix: str, search_only: bool) -> None:
    sb, bucket = client_and_bucket()
    store = sb.storage.from_(bucket)
    manifest = remote_manifest(store, prefix)
    files = manifest["files"]
    if not files:
        sys.exit(f"원격 manifest 없음: {bucket}/{prefix}/{MANIFEST}")

    targets = [r for r in files if (not search_only or r in SEARCH_FILES)]
    got = 0
    for rel in targets:
        dst = root / rel
        if dst.exists() and sha256(dst) == files[rel]["sha256"]:
            continue                               # 캐시 유효
        dst.parent.mkdir(parents=True, exist_ok=True)
        data = store.download(f"{prefix}/{rel}")
        if hashlib.sha256(data).hexdigest() != files[rel]["sha256"]:
            sys.exit(f"해시 불일치: {rel}")
        dst.write_bytes(data)
        got += 1
        print(f"  ↓ {rel}")

    # 페이지를 열 때 책 검증에 쓰도록 manifest도 로컬에 저장
    root.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] 다운로드 {got}개 / 대상 {len(targets)}개 → {root} (+{MANIFEST})")


def verify_book(book: Path, manifest_path: Path) -> dict:
    """로컬 PDF가 데이터셋 원본과 같은지 확인. 서버 시작 시 1번 호출."""
    exp = json.loads(manifest_path.read_text("utf-8")).get("book")
    if not exp:
        raise RuntimeError("manifest에 book 정보가 없습니다. upload를 다시 실행하세요.")
    if sha256(book) != exp["sha256"]:
        raise RuntimeError(f"로컬 PDF({book})가 데이터셋 원본({exp['name']})과 다릅니다.")
    return exp


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["upload", "download"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--book", default="book/book.pdf", help="데이터셋을 만든 원본 PDF (upload 시 해시 기록)")
    ap.add_argument("--prefix", default="book1", help="교재 단위 원격 폴더명")
    ap.add_argument("--search-only", action="store_true", help="검색용 파일만 받기")
    a = ap.parse_args()
    if a.cmd == "upload":
        upload(Path(a.root or "dataset"), a.prefix, Path(a.book))
    else:
        download(Path(a.root or "dataset_cache"), a.prefix, a.search_only)
