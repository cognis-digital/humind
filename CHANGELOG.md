# Changelog

## [0.3.0] — 2026-06-13

The "thinks better" release — humind now **learns from outcomes**, **predicts**, and
**reasons about structure**, grounded in named mechanisms from RL, neurobiology, and
systems dynamics. All additions are parallel layers; the v0.2 API is unchanged and the
core stays pure-stdlib and explainable.

### Added
- **Reinforcement learning** (`learning.py`):
  - `ValueLearner` — **TD(lambda)** credit assignment with **eligibility traces**
    (Sutton & Barto). `Mind.reinforce(reward)` flows credit back to recently-attended
    context, so the engine *discovers which context is worth attending to*.
  - `LexiconLearner` — online affect by the **Rescorla-Wagner / delta rule**: words in a
    reinforced episode acquire valence from experience (an unseen domain term like
    "shadowfleet" becomes "threatening" after negative outcomes — no hand-built lexicon).
- **Neurobiology-inspired memory** (`memory.py`):
  - `AssociativeMemory` — **Hebbian** co-occurrence network + **spreading activation**
    (Collins & Loftus / ACT-R). `Mind.predict(cue)` is cued recall.
  - `CognitiveMemory.consolidate()` — **Complementary Learning Systems**: recurring
    episodic entities distil into durable semantic facts (fast hippocampal -> slow
    neocortical).
- **Predictive coding** — `ContextFrame.surprise` = prediction error vs the prior; novel
  input is precision-weighted to grab more attention (surprise drops as input repeats).
- **Systems thinking** (`systems.py`): a **causal-loop diagram** built from causal
  language (`causal_links()` extracts polarity-signed cause->effect from "leads to",
  "reduces", "because of", ...). `feedback_loops()` classifies **reinforcing (R)** vs
  **balancing (B)** loops; `leverage_points()` ranks intervention nodes by centrality
  (Forrester / Sterman / Meadows).
- `Mind.priorities()` — value-weighted attention blending working-memory activation,
  learned value, and causal centrality. Negation-aware affect ("not safe" -> negative).
- CLI: `humind causal ...` (links + R/B loops + leverage), `humind learn` (RL demo),
  `humind bench` (throughput, ~3.7k frames/sec stdlib); `think` now shows
  priorities/prediction.
- 15 new tests (29 total) covering every mechanism + the new CLI commands.

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
