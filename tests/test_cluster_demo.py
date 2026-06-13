"""The executable interop demo runs end-to-end (maritimeint→humind→agentlex)."""

from __future__ import annotations

import importlib.util
import os


def _load_demo():
    p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "cluster_demo.py")
    spec = importlib.util.spec_from_file_location("cluster_demo", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cluster_demo_chains_to_escalation(capsys):
    m = _load_demo()
    rc = m.main()                       # maritimeint (or fallback) -> humind -> agentlex
    out = capsys.readouterr().out
    assert rc == 0
    assert "humind->agentlex" in out and "observed(neptune-star, high)" in out
    # the agentlex rule fires on high-risk AND sanctioned
    assert "escalate(neptune-star)" in out
