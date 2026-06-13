# Changelog

## [0.2.0] — 2026-06-13
### Added
- **Optional LLM enrichment** (`addins.py`): `Mind.perceive(text, enrich=True)` adds a
  concise analyst interpretation as a `notes` annotation when a model backend is
  reachable — set `HUMIND_ENDPOINT` / `OPENAI_BASE_URL`, pass `--endpoint`, or let it
  auto-discover one (edgemesh gateway / your fleet / Ollama / vLLM / …). **Graceful:**
  no backend → enrichment is skipped and the transparent stdlib core is unaffected.
- `ContextFrame.notes` field; `humind perceive --ai [--endpoint --model]`.
- Tests for discovery gating + interpret/perceive against a live mock `/v1` (13 total).

## [0.1.0] — 2026-06-13
- Initial release: NL context extraction (entities/intent/affect/salience), working/
  episodic/semantic memory, and the `agentlex` tandem (`express`/`ingest`). Stdlib core.
