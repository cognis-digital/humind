# Changelog

## [Unreleased]

### Added
- **`Mind.introspect(n=5)`** — a single JSON-serialisable snapshot of the whole
  cognitive state (focus, blended priorities, learned values, associative cues,
  durable facts, learned valences, feedback loops + leverage, last surprise, and
  counters). Built-in types only, so it drops straight into `json.dumps`, a log line,
  or a dashboard without a custom encoder.
- **`humind explain "…" "…"`** — CLI wrapper that perceives a sequence and prints the
  `introspect()` snapshot as indented JSON (`-n` bounds each ranked section).
- Demo 21 (`introspection_snapshot`) and `tests/test_introspect.py`.

## [0.3.2] — 2026-07-01

The "hardening" release — clearer errors, fixed edge-case bugs, and a much deeper
test/demo corpus. The public API is unchanged.

### Fixed
- **`CausalGraph.feedback_loops()` collapsed distinct directed cycles.** Deduplication
  keyed on `frozenset(cycle)` (the *set* of nodes), so two genuinely different loops over
  the same concepts — different direction or different link polarities, e.g. a
  reinforcing `a→b→c→a` and a balancing `a→c→b→a` — were merged into one and the second
  was silently dropped. Deduplication is now by the **canonical rotation of the directed
  cycle**, so each distinct loop is reported once with its correct R/B classification.
- **`WorkingMemory` silently discarded items on a non-positive capacity.** `capacity=0`
  made `del self.items[0:]` wipe the buffer on every `add` (a silent no-op store), and a
  negative capacity behaved erratically. Construction now validates `capacity ≥ 1`,
  `0 < decay < 1` (so activation actually forgets), and `threshold ≥ 0`.
- **`WorkingMemory.add()` left a stale item order after re-activation.** Re-activating an
  existing item bumped its activation but skipped the sort/evict step; it now keeps the
  buffer consistently ordered and capacity-bounded on every path.

### Changed (hardening — API stable)
- Text perception (`extract`, `classify_intent`, `entities`, `affect`, `causal_links`,
  `salient`) now raises a clear `TypeError` naming the argument for non-`str` input,
  instead of leaking an `AttributeError` from deep in the pipeline. `salient(k=…)`
  validates `k ≥ 0`.
- `ValueLearner` / `LexiconLearner` validate their learning-rate / discount / trace
  parameters at construction, and reject non-finite rewards (NaN/inf would silently
  poison learned values). `EpisodicMemory.recent(n)` and `AssociativeMemory(cap=…)`
  validate their arguments.

### Added
- **223 new tests (262 total)** across five new edge/error-path suites plus expanded
  CLI/connect/addins coverage: convergence and extreme parameters for the learners,
  memory dynamics and consolidation idempotence, causal-loop detection and the directed-
  cycle fix, negation windows and affect boundaries, intent precedence, malformed input,
  and graceful-degradation paths.
- **15 new runnable demos (20 total)** in `demos/` (deep-dives 6–17 and capstones
  18–20), all wired into `run_all.py`, `tests/test_demos.py`, and `docs/DEMOS.md`; each
  exits 0 under `PYTHONUTF8=1`.

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
