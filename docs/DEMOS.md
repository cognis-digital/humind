# Demos

Five runnable scenarios in [`../demos/`](../demos/), each targeting a different
audience and a different layer of the toolkit. Every scenario builds its own fresh
`Mind` from a small, **bundled inline transcript** — no network, no external data,
no shared state — so you can run them in any order or on their own.

```bash
PYTHONUTF8=1 python demos/run_all.py            # all five, end to end (exits 0)
PYTHONUTF8=1 python demos/02_reinforcement_learning.py    # or just one
```

> `PYTHONUTF8=1` is only needed on Windows consoles that aren't already UTF-8.

## 1. Perception pipeline — *one transparent frame per utterance*
**Audience:** ML engineers.
Runs the transcript through `extract()` and shows the `ContextFrame` for each line
— intent, valence/arousal, entities, salient terms, causal links — then proves the
pieces: negation flips affect ("not safe" → negative), causal language is parsed
into `(cause, effect, polarity)` triples, and every field traces to an inspectable
lexicon/heuristic. No weights, fully reproducible.

## 2. Reinforcement learning — *credit the context that preceded the outcome*
**Audience:** RL researchers.
The headline learning story. An unseen domain word (`shadowfleet`) starts with zero
affect; after repeatedly appearing before bad outcomes (`reinforce(-1.0)`), the
**Rescorla-Wagner** delta rule drives its learned valence negative until perception
flags it on its own, while **TD(λ) eligibility traces** assign value back to the
attended context. Then `priorities()` re-ranks attention by blending activation,
learned value, and causal centrality. Nothing was hand-labelled.

## 3. Memory & association — *four stores, surprise, and consolidation*
**Audience:** students & educators.
Opens each store side by side: the decaying **working** focus, the **episodic**
timeline (with predictive-coding `surprise` per frame), durable **semantic** facts,
and the **associative** Hebbian network — where `spread()` / `predict()` show
cueing one concept bringing associates to mind. Then **CLS `consolidate()`** distils
a recurring entity into a durable fact, and a side-by-side shows a repeat is
unsurprising while a new contact is maximally surprising.

## 4. Systems thinking — *causal-loop diagram, feedback loops, leverage*
**Audience:** systems builders.
Feeds the mind plain-English causal statements and lets the structure fall out: a
**causal-loop diagram** with edge polarities, a discovered **reinforcing (R) loop**
(darkness → escalation → sanctions → smuggling → darkness), the **balancing** lever
that breaks it, and the **leverage points** ranked by causal centrality (Forrester /
Sterman / Meadows). Behaviour comes from loop structure, derived from prose.

## 5. Two minds, one language — *the agentlex tandem + inference*
**Audience:** systems builders (multi-agent).
A scout `perceive`s free text, `express`es it as a precise agentlex message
(`observed(neptune-star, high)`), and a command mind `ingest`s it across the wire —
unifiable symbols, not prose. Then an agentlex `KnowledgeBase` **derives**
escalations from a rule (`escalate(?v) :- observed(?v, high), sanctioned(?v)`), and
an optional private-LLM brief is added when a backend is reachable, skipped
gracefully otherwise.

---

Each demo prints clear, narrated output and exits 0, so they double as smoke tests
— `tests/test_demos.py` runs every scenario and `run_all` under `pytest`.
