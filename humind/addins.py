"""Optional LLM enrichment for humind — augment the transparent core, never replace it.

The stdlib extractor always runs and is explainable. When a model backend is
reachable (set `HUMIND_ENDPOINT` / `OPENAI_BASE_URL`, or let it auto-discover one on
the common local ports — an edgemesh gateway, your fleet, Ollama, vLLM, …), `interpret`
adds a concise analyst reading of an utterance as a `notes` annotation on the frame.

Availability is deployment-limited: no backend → enrichment is skipped, core unaffected.
Pure standard library (urllib).
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable

ENV_ENDPOINT = os.environ.get("HUMIND_ENDPOINT") or os.environ.get("OPENAI_BASE_URL")
COMMON_PORTS = [8780, 11434, 8000, 8080, 1234, 1337, 30000, 8774, 8773]


def probe(base_url: str, timeout: float = 2.0) -> list[str] | None:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except Exception:
        return None
    items = data.get("data", data) if isinstance(data, dict) else data
    return [it.get("id") if isinstance(it, dict) else it for it in (items or []) if it]


def discover(probe_fn: Callable[[str], list[str] | None] = probe) -> tuple[str, list[str]] | None:
    """Return (base_url, models) for the first reachable backend, or None."""
    if ENV_ENDPOINT:
        m = probe_fn(ENV_ENDPOINT)
        if m is not None:
            return ENV_ENDPOINT, m
    for port in COMMON_PORTS:
        url = f"http://127.0.0.1:{port}"
        m = probe_fn(url)
        if m is not None:
            return url, m
    return None


def chat(base_url: str, model: str, messages: list[dict], *, timeout: float = 60.0) -> str:
    body = json.dumps({"model": model, "messages": messages}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))["choices"][0]["message"]["content"]


def interpret(text: str, base_url: str, model: str, **kw) -> str:
    """A concise analyst reading of an utterance (one or two sentences)."""
    prompt = ("Give a one-to-two sentence analyst interpretation of this message: what "
              "is the speaker conveying, and any implied context. Be concise and literal.\n\n"
              f"MESSAGE: {text}")
    return chat(base_url, model, [{"role": "user", "content": prompt}], **kw).strip()
