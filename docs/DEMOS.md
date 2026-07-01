# Demos

Twenty runnable scenarios in [`../demos/`](../demos/), each targeting a different
audience and a different layer of the toolkit. Every scenario builds its own fresh
`Mind` (or raw component) from a small, **bundled inline transcript** — no network,
no external data, no shared state — so you can run them in any order or on their own.

```bash
PYTHONUTF8=1 python demos/run_all.py            # all twenty, end to end (exits 0)
PYTHONUTF8=1 python demos/02_reinforcement_learning.py    # or just one
```

Demos 1–5 are the guided tour of each layer; 6–17 are focused deep-dives into a
single mechanism; 18–20 are capstones (the full loop, multi-agent scale, and the
error contract).

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

## 6. Affect & negation — *a valence word flips within a 2-token window*
**Audience:** linguists / analysts.
Walks minimal pairs ("the corridor is safe" vs "…is not safe") to show exactly
where the negation flip fires and where it doesn't (a negator 3+ tokens away can't
reach the word), and that arousal (urgency) is counted independently of valence.

## 7. Intent / speech acts — *what the speaker is DOING*
**Audience:** dialogue-systems builders.
Classifies a mini-dialogue into `inform / query / request / propose / agree /
refuse`, then shows the fixed precedence on ambiguous lines (a refusal inside a
question is still a query). Intent lines up 1:1 with an agentlex performative.

## 8. Working memory — *a small, decaying attention buffer*
**Audience:** cognitive-science students.
Drives a capacity-3 buffer tick by tick with an activation bar: contacts enter,
a fourth forces an eviction, decay pulls everything down, a rehearsed item jumps
back to the top, and stale items fall below threshold and leave focus (ACT-R / Miller).

## 9. Spreading activation — *cue one concept, associates come to mind*
**Audience:** memory researchers.
Builds the Hebbian co-occurrence network from a short transcript and probes it with
cues; repeated co-occurrence ranks above a single one. `Mind.predict()` uses the
same mechanism (Collins & Loftus), and the weights are inspectable.

## 10. Predictive coding — *surprise is the fraction you did not expect*
**Audience:** computational-neuroscience readers.
Streams a watch log and prints each line's `surprise` (prediction error against a
running expectation): an exact repeat is quiet, a brand-new contact spikes, and its
follow-up is only partly novel once it's been associated.

## 11. TD(λ) credit assignment — *reward flows back along the trace*
**Audience:** RL researchers (deep dive).
Attends a 5-step episode one cue at a time, delivers a single terminal reward, and
prints how much value each earlier cue received — recency-weighted by the
eligibility trace (decay `γ·λ` per step). The earliest precursor is credited, but least.

## 12. Online valence learning — *words acquire affect from outcomes*
**Audience:** learning-theory readers.
Tracks two unknown words in parallel — one preceding bad outcomes, one good —
and prints the Rescorla-Wagner learning curve as each converges (monotone, bounded
to `[-1, 1]`) and crosses the `|0.1|` confidence gate into `overrides()`.

## 13. CLS consolidation — *recurring episodes become durable facts*
**Audience:** memory-systems readers.
Streams a log where one contact recurs and others are one-offs, tallies entity
recurrence, then runs `consolidate(min_count=2)` — only the recurring entity is
promoted from the fast episodic log into slow semantic memory — and shows it's idempotent.

## 14. Feedback-loop polarity — *R (amplifying) vs B (self-correcting)*
**Audience:** system-dynamics readers (deep dive).
Three hand-built loops make the rule unambiguous: an all-positive spiral (0
negatives → R), a thermostat (1 negative → B), and a double-negative pair (2 → R).
Polarity is purely structural — count the negative links around the loop.

## 15. Leverage points — *intervene where the structure is most connected*
**Audience:** policy / analysis readers.
Assembles a smuggling dynamic from prose, ranks concepts by causal centrality, and
shows why the hub the balancing levers attach to (`smuggling`) outranks the visible
symptom (`darkness`) — the place to push is not always the place you see (Meadows).

## 16. Value-weighted priorities — *activation + learned value + centrality*
**Audience:** decision-support builders.
Rewards one thread of context, then shows how `priorities()` diverges from raw
attention: proven-valuable and structurally central context is lifted above loud-
but-unproven recency, so "routine chatter" can't crowd out what matters.

## 17. Causal extraction — *prose → (cause, effect, polarity) triples*
**Audience:** NLP / knowledge-graph builders.
Runs a batch of sentences through the connective parser — positive (`leads to`),
negative (`reduces`), and reversed (`because of`, which puts the effect first and is
normalised back) — and prints each triple, plus a non-causal line that yields nothing.

## 18. Full cognitive loop — *perceive → learn → consolidate → reason*
**Audience:** integrators / evaluators.
A capstone driving one `Mind` through the entire cycle on a single evolving
scenario: perception building every store, surprise falling, a terminal reward
teaching TD + valence, consolidation, and finally blended priorities, prediction,
loops and leverage — the integration test as narrative.

## 19. Multi-agent broadcast — *many scouts express, one command ingests*
**Audience:** multi-agent-systems builders.
Scales the tandem past two: four scout minds each `express` a local report onto one
agentlex channel; a command mind `ingest`s the whole broadcast into shared memory,
and a rule derives escalations from the merged facts. Degrades cleanly without agentlex.

## 20. Robustness & errors — *loud on bugs, quiet on missing infrastructure*
**Audience:** operators / integrators.
Shows both sides of the error contract: malformed / out-of-range inputs (non-str
text, non-forgetting decay, non-finite reward, bad learning rate) raise clear,
specific exceptions, while enrichment against an unreachable backend is skipped
without disturbing the core frame. No traceback escapes.

---

Each demo prints clear, narrated output and exits 0, so they double as smoke tests
— `tests/test_demos.py` runs every scenario and `run_all` under `pytest`.
