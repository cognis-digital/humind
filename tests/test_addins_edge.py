"""Edge cases for optional LLM enrichment (humind.addins).

Discovery gating (env + port scan), probe parsing of both {"data":[...]} and bare
list model listings, chat POST round-trip, and the guarantee that a failed
enrichment never breaks the transparent core frame.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from humind import addins
from humind.mind import Mind


class _Mock(BaseHTTPRequestHandler):
    LIST_STYLE = "data"                              # or "bare"

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            if _Mock.LIST_STYLE == "data":
                self._send({"data": [{"id": "m1"}, {"id": "m2"}]})
            else:
                self._send(["m1", "m2"])
        else:
            self._send({}, 404)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._send({"choices": [{"message": {"content": "concise reading"}}]})


@pytest.fixture()
def gateway():
    _Mock.LIST_STYLE = "data"
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Mock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield "http://127.0.0.1:%d" % srv.server_address[1]
    srv.shutdown()


# --- probe / discover --------------------------------------------------------
def test_probe_parses_data_listing(gateway):
    assert addins.probe(gateway) == ["m1", "m2"]


def test_probe_parses_bare_list(gateway):
    _Mock.LIST_STYLE = "bare"
    assert addins.probe(gateway) == ["m1", "m2"]


def test_probe_unreachable_returns_none():
    assert addins.probe("http://127.0.0.1:1", timeout=0.2) is None


def test_discover_none_when_all_probes_fail():
    assert addins.discover(probe_fn=lambda url: None) is None


def test_discover_returns_first_reachable():
    # a probe_fn that only answers a specific url
    def probe_fn(url):
        return ["only"] if url.endswith(":11434") else None
    where = addins.discover(probe_fn=probe_fn)
    assert where is not None and where[1] == ["only"]


# --- chat / interpret --------------------------------------------------------
def test_chat_roundtrip(gateway):
    out = addins.chat(gateway, "m1", [{"role": "user", "content": "hi"}])
    assert out == "concise reading"


def test_interpret_returns_stripped_text(gateway):
    out = addins.interpret("vessel went dark", gateway, "m1")
    assert out == "concise reading"


# --- perceive enrichment is best-effort --------------------------------------
def test_perceive_enrich_sets_notes(gateway):
    f = Mind("s").perceive("vessel NEPTUNE-STAR went dark",
                           enrich=True, endpoint=gateway, model="m1")
    assert f.notes == "concise reading"
    assert "NEPTUNE-STAR" in f.entities            # core still ran


def test_perceive_enrich_failure_is_graceful():
    f = Mind("s").perceive("vessel went dark", enrich=True, endpoint="http://127.0.0.1:1")
    assert f.notes == "" and f.intent == "inform"   # core intact, no exception


def test_perceive_without_enrich_never_calls_backend():
    # no endpoint, enrich False -> notes stays empty, purely offline
    f = Mind("s").perceive("vessel went dark")
    assert f.notes == ""
