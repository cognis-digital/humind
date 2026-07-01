"""Edge cases for the cognis-connect emit shim (humind.connect).

These do NOT require the optional cognis-connect dependency: they cover the
tool-side mapping (map_record) and the graceful failure when the dependency is
absent, so the shim is exercised in a clean CI without the extra installed.
"""

from __future__ import annotations

import io
import sys

from humind import connect


def test_map_record_extracts_title_from_content():
    out = connect.map_record({"content": "Vessel dark\nmore detail", "severity": "high"})
    assert out["title"] == "Vessel dark"
    assert out["severity"] == "high" and out["description"].startswith("Vessel dark")


def test_map_record_defaults_when_fields_missing():
    out = connect.map_record({})
    assert out["title"] == "" and out["severity"] == "info" and out["type"] == "unknown"
    assert out["tags"] == []


def test_map_record_passes_indicator_fields():
    rec = {"content": "x", "ipv4": "203.0.113.5", "domain": "evil.example",
           "sha256": "ab" * 32, "cve": "CVE-2024-0001"}
    out = connect.map_record(rec)
    assert out["ipv4"] == "203.0.113.5" and out["domain"] == "evil.example"
    assert out["cve"] == "CVE-2024-0001"


def test_map_record_non_dict_returns_input_unchanged():
    # the guarded except-branch returns the record as-is on any error
    sentinel = object()
    assert connect.map_record(sentinel) is sentinel


def test_source_constant():
    assert connect.SOURCE == "humind"


def test_emit_without_dependency_reports_cleanly(monkeypatch, capsys):
    # simulate cognis-connect not being installed
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("cognis_connect"):
            raise ImportError("no cognis-connect")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.stdin = io.StringIO('[{"content": "x"}]')
    try:
        rc = connect.emit_main(["--to", "sigma"])
    finally:
        sys.stdin = sys.__stdin__
    assert rc == 1
    assert "needs cognis-connect" in capsys.readouterr().err
