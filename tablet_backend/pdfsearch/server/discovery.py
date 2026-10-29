"""같은 와이파이의 앱이 서버 주소를 스스로 찾게 해 주는 UDP 응답기.

앱이 브로드캐스트로 b"YOONS_DISCOVER" 를 UDP 8002 번으로 보내면
{"service": "yoonsimage", "port": 8001} 을 보낸 곳으로 답한다.
앱은 답이 온 IP + port 로 HTTP 서버 주소를 만든다 (서버가 자기 IP 를 몰라도 됨).
"""
from __future__ import annotations

import json
import logging
import socket
import threading

log = logging.getLogger("server.discovery")

MAGIC = b"YOONS_DISCOVER"
SERVICE = "yoonsimage"


class DiscoveryResponder:
    def __init__(self, http_port: int, udp_port: int = 8002):
        self.http_port = http_port
        self.udp_port = udp_port
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", self.udp_port))
            s.settimeout(0.5)
        except OSError as e:
            log.warning("자동 찾기(UDP %d) 시작 실패: %s -> 앱에서 주소를 직접 입력해야 함", self.udp_port, e)
            return False
        self._sock = s
        self._thread = threading.Thread(target=self._run, name="discovery", daemon=True)
        self._thread.start()
        log.info("자동 찾기 응답 대기: UDP %d", self.udp_port)
        return True

    def _run(self) -> None:
        reply = json.dumps({"service": SERVICE, "port": self.http_port}).encode()
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                break
            if data.strip() == MAGIC:
                try:
                    self._sock.sendto(reply, addr)
                except OSError:
                    pass

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=2)


def lan_addresses() -> list[str]:
    """이 PC 의 내부 IPv4 주소들 (시작 로그에 '직접 입력용 주소'로 보여 줌)."""
    found: list[str] = []
    try:                                    # 기본 경로 인터페이스 (패킷은 실제로 안 나감)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError:
        pass
    return found
