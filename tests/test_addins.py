"""Optional LLM enrichment: discovery gating + interpret/perceive against a live mock."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from humind import addins
from humind.mind import Mind


class _MockOAI(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self._send({"data": [{"id": "test-model"}]}) if self.path.rstrip("/").endswith("/v1/models") else self._send({}, 404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"choices": [{"message": {"content": "The scout reports a vessel has gone dark; implies evasion."}}]})


@pytest.fixture()
def gateway():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _MockOAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


def test_discover_none_when_unreachable():
    assert addins.discover(probe_fn=lambda url: None) is None


def test_interpret_against_live_mock(gateway):
    out = addins.interpret("vessel went dark", gateway, "test-model")
    assert "gone dark" in out


def test_perceive_enrich_sets_notes(gateway):
    frame = Mind("scout").perceive("vessel NEPTUNE-STAR went dark", enrich=True, endpoint=gateway, model="test-model")
    assert frame.notes and "evasion" in frame.notes
    assert "NEPTUNE-STAR" in frame.entities      # core extraction still ran


def test_perceive_without_backend_is_graceful():
    # bad endpoint -> enrichment skipped, no exception, core frame intact
    frame = Mind("scout").perceive("vessel went dark", enrich=True, endpoint="http://127.0.0.1:1")
    assert frame.notes == "" and frame.intent == "inform"
