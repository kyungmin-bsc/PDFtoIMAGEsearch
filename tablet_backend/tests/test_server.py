"""서버 흐름 테스트 (모델 없이). 실행: tablet_backend/ 에서  python -m pytest tests -q

검출기와 DINOv2 를 가벼운 가짜로 바꿔 끼워서
업로드 -> 처리 -> ready -> 검색 -> 그림 이미지 까지 API 규격대로 동작하는지 본다.
(torch 가 없는 환경이면 features.py import 용 최소 가짜 torch 를 넣는다)
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
except ImportError:                                    # features.py 를 import 만 할 수 있게
    fake = types.ModuleType("torch")
    fake.inference_mode = lambda: (lambda f: f)
    fake.__version__ = "fake"
    nn = types.ModuleType("torch.nn")
    fn = types.ModuleType("torch.nn.functional")
    nn.functional = fn
    fake.nn = nn
    sys.modules.update({"torch": fake, "torch.nn": nn, "torch.nn.functional": fn})

from fastapi.testclient import TestClient  # noqa: E402

from pdfsearch.common.schema import Box  # noqa: E402
from pdfsearch.embedding.features import normalize  # noqa: E402
from pdfsearch.server.api import create_app  # noqa: E402
from pdfsearch.server.store import Engines, default_config  # noqa: E402

PDF = ROOT / "book" / "circuit_test_book.pdf"


class FakeDetector:
    """홀수 쪽마다 페이지 위쪽 절반에 그림 하나가 있다고 답한다."""
    weights_hash = "fake"

    def detect(self, item):
        if item.pdf_index % 2:
            return []
        W, H = item.size_px
        return [Box((int(W * .15), int(H * .1), int(W * .85), int(H * .5)), "figure", 0.95)]


class FakeEmbedder:
    """실제 normalize() 결과를 16x16 으로 줄여 벡터로 쓴다 (같은 그림 -> 높은 유사도)."""
    dim = 256

    def embed_images(self, rgbs, mode, batch=16):
        if not rgbs:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for r in rgbs:
            g = cv2.resize(normalize(r, mode), (16, 16), interpolation=cv2.INTER_AREA)
            v = (255 - g.astype(np.float32)).ravel()
            out.append(v / (np.linalg.norm(v) + 1e-6))
        return np.stack(out)


@pytest.fixture()
def client(tmp_path):
    engines = Engines(default_config(),
                      detector_factory=lambda cfg: FakeDetector(),
                      embedder_factory=FakeEmbedder)
    with TestClient(create_app(tmp_path, engines, mode="edge", discovery=False)) as c:
        yield c


def _wait_ready(c, doc_id, timeout=120):
    t = time.time()
    while time.time() - t < timeout:
        st = c.get(f"/api/v1/documents/{doc_id}").json()
        if st["status"] != "processing":
            return st
        time.sleep(0.2)
    raise TimeoutError


def test_full_flow(client):
    pdf = PDF.read_bytes()
    r = client.post("/api/v1/documents", files={"file": ("교재.pdf", pdf, "application/pdf")})
    assert r.status_code == 202, r.text
    doc = r.json()
    assert doc["status"] == "processing" and doc["filename"] == "교재.pdf"

    st = _wait_ready(client, doc["document_id"])
    assert st["status"] == "ready", st
    assert st["page_count"] > 0 and st["figure_count"] > 0

    # 같은 PDF 재업로드 -> 처리 없이 바로 ready (200)
    r2 = client.post("/api/v1/documents", files={"file": ("교재.pdf", pdf, "application/pdf")})
    assert r2.status_code == 200 and r2.json()["status"] == "ready"

    # 저장된 그림 하나를 그대로 질의로 쓰면 1위여야 한다
    img = client.get(f"/api/v1/documents/{doc['document_id']}/figures/p0003_f01/image?size=full")
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    s = client.post(f"/api/v1/documents/{doc['document_id']}/search",
                    files={"sketch": ("sketch.png", img.content, "image/png")},
                    data={"top_k": "3"})
    assert s.status_code == 200, s.text
    res = s.json()["results"]
    assert 1 <= len(res) <= 3
    top = res[0]
    assert top["page"] == 3 and top["figure_id"].startswith("p0003")
    assert 0 <= top["score"] <= 1
    x, y, w, h = top["bbox"]
    assert 0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
    assert abs(x - .15) < .03 and abs(y - .1) < .03          # 가짜 검출기의 위치 그대로

    thumb = client.get(top["image_url"])
    assert thumb.status_code == 200
    th = cv2.imdecode(np.frombuffer(thumb.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(th.shape[:2]) <= 512


def test_errors(client):
    r = client.post("/api/v1/documents", files={"file": ("a.pdf", b"hello", "application/pdf")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_FILE"

    r = client.get("/api/v1/documents/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "DOCUMENT_NOT_FOUND"

    doc = client.post("/api/v1/documents",
                      files={"file": ("b.pdf", PDF.read_bytes(), "application/pdf")}).json()
    _wait_ready(client, doc["document_id"])
    blank = cv2.imencode(".png", np.full((64, 64, 3), 255, np.uint8))[1].tobytes()
    r = client.post(f"/api/v1/documents/{doc['document_id']}/search",
                    files={"sketch": ("s.png", blank, "image/png")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "EMPTY_SKETCH"


def test_cors_for_flutter_web(client):
    """Chrome 에서 띄운 Flutter(다른 포트)도 응답을 읽을 수 있어야 한다."""
    r = client.get("/api/v1/health", headers={"Origin": "http://localhost:53000"})
    assert r.headers.get("access-control-allow-origin") == "*"
    pre = client.options("/api/v1/documents", headers={
        "Origin": "http://localhost:53000", "Access-Control-Request-Method": "POST"})
    assert pre.status_code == 200


def test_list_and_delete(client, tmp_path):
    doc = client.post("/api/v1/documents",
                      files={"file": ("c.pdf", PDF.read_bytes(), "application/pdf")}).json()
    _wait_ready(client, doc["document_id"])

    docs = client.get("/api/v1/documents").json()["documents"]
    assert [d["document_id"] for d in docs] == [doc["document_id"]]
    assert docs[0]["size_bytes"] > PDF.stat().st_size and docs[0]["created_at"]

    r = client.delete(f"/api/v1/documents/{doc['document_id']}")
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert client.get("/api/v1/documents").json()["documents"] == []
    assert client.get(f"/api/v1/documents/{doc['document_id']}").status_code == 404
    assert not (tmp_path / "documents" / doc["document_id"]).exists()
    assert client.delete(f"/api/v1/documents/{doc['document_id']}").status_code == 404

    # 지운 뒤 다시 올리면 처음부터 다시 처리
    r = client.post("/api/v1/documents", files={"file": ("c.pdf", PDF.read_bytes(), "application/pdf")})
    assert r.status_code == 202


def test_discovery_responder():
    """앱이 보내는 것과 같은 UDP 질문에 서버 포트를 답한다."""
    import json
    import socket
    from pdfsearch.server.discovery import DiscoveryResponder

    r = DiscoveryResponder(http_port=8001, udp_port=18002)
    assert r.start()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(2)
            s.sendto(b"hello", ("127.0.0.1", 18002))          # 엉뚱한 메시지는 무시
            s.sendto(b"YOONS_DISCOVER", ("127.0.0.1", 18002))
            data, _ = s.recvfrom(1024)
        assert json.loads(data) == {"service": "yoonsimage", "port": 8001}
    finally:
        r.stop()
