"""Context extraction from natural language — the 'perception' stage of humind.

Pulls the human-meaningful structure out of an utterance, stdlib-only and
explainable: entities, the speech-act **intent**, **affect** (valence/arousal), and
the **salient** terms. An optional LLM add-in (see `humind.addins`, pointed at an
edgemesh gateway / your fleet) can enrich this — but the core always works offline.

This is deliberately transparent (lexicons + heuristics), not a black box — so a
downstream agent can see *why* a piece of context was extracted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

_QUESTION = re.compile(r"\?\s*$")
_REQUEST = re.compile(r"\b(please|could you|can you|would you|let's|i need|we need)\b", re.I)
_PROPOSE = re.compile(r"\b(i (think|suggest|propose|believe)|maybe|perhaps|we should)\b", re.I)
_AGREE = re.compile(r"\b(yes|agreed|sounds good|confirmed|affirmative|ok|okay)\b", re.I)
_REFUSE = re.compile(r"\b(no|negative|decline|won'?t|cannot|refuse)\b", re.I)
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z0-9]+(?:[ -][A-Z0-9][a-zA-Z0-9]+)*)\b")
_NUMID = re.compile(r"\b([A-Za-z]*-?\d{3,}[A-Za-z0-9-]*)\b")

# tiny affect lexicon (valence): word -> score in [-1, 1]
_VALENCE = {
    "urgent": -0.6, "critical": -0.8, "risk": -0.5, "threat": -0.7, "danger": -0.8,
    "fail": -0.6, "lost": -0.5, "good": 0.6, "safe": 0.6, "clear": 0.4, "success": 0.7,
    "confirmed": 0.4, "high": -0.4, "low": 0.3, "calm": 0.5, "alarm": -0.7,
}
_AROUSAL = {"urgent", "critical", "danger", "threat", "alarm", "now", "immediately", "emergency"}
_NEGATORS = {"not", "no", "never", "without", "cannot", "can't", "isn't", "aren't",
             "wasn't", "won't", "don't", "doesn't", "lacks", "lack"}
_STOP = set("a an the of to in on at is are was were be been and or but for with this that "
            "it its as by from we i you he she they them our your".split())
# all-caps status/severity markers that read as entities but aren't named things
_NONENTITY = set("critical urgent alert warning notice note high low medium fyi update status "
                 "emergency caution danger info report flash priority routine".split())

# causal connectives -> (polarity, reversed?)  reversed means 'effect <conn> cause'
_CAUSAL = [
    (r"causes|leads to|results in|drives|triggers|increases|boosts|enables|raises|escalates", 1, False),
    (r"because of|due to|driven by|caused by|result of", 1, True),
    (r"reduces|decreases|prevents|mitigates|limits|suppresses|lowers|slows|dampens", -1, False),
]
_CAUSAL = [(re.compile(r"(.{0,60}?)\b(?:" + pat + r")\b(.{0,60})", re.I), pol, rev)
           for pat, pol, rev in _CAUSAL]


@dataclass
class ContextFrame:
    text: str
    intent: str                              # inform | request | query | propose | agree | refuse
    entities: list[str] = field(default_factory=list)
    salient: list[str] = field(default_factory=list)
    valence: float = 0.0                     # negative = threatening/bad, positive = good
    arousal: float = 0.0                     # 0..1 urgency
    causes: list[tuple[str, str, int]] = field(default_factory=list)  # (cause, effect, polarity)
    surprise: float = 0.0                    # prediction error vs current expectation (set by Mind)
    notes: str = ""                          # optional LLM-enrichment annotation (see addins)

    def to_dict(self) -> dict:
        return asdict(self)


def _as_text(text: object, arg: str = "text") -> str:
    """Coerce/validate a text argument, with a clear error instead of a leaked
    AttributeError deep in the pipeline."""
    if not isinstance(text, str):
        raise TypeError(f"{arg} must be a str, got {type(text).__name__}")
    return text


def classify_intent(text: str) -> str:
    text = _as_text(text)
    if _QUESTION.search(text):
        return "query"
    if _REFUSE.search(text):
        return "refuse"
    if _AGREE.search(text):
        return "agree"
    if _REQUEST.search(text):
        return "request"
    if _PROPOSE.search(text):
        return "propose"
    return "inform"


def entities(text: str) -> list[str]:
    text = _as_text(text)
    found = []
    for m in _NUMID.finditer(text):
        found.append(m.group(1))
    for m in _ENTITY.finditer(text):
        e = m.group(1)
        if e.lower() not in _STOP and e.lower() not in _NONENTITY:
            found.append(e)
    # de-dupe, preserve order
    seen, out = set(), []
    for e in found:
        if e not in seen:
            seen.add(e); out.append(e)
    return out


def affect(text: str, extra: dict | None = None) -> tuple[float, float]:
    """Valence/arousal. Negation within the prior two tokens flips a valence word.
    `extra` supplies learned word-valences (see humind.learning.LexiconLearner)."""
    text = _as_text(text)
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0, 0.0
    lex = _VALENCE if not extra else {**_VALENCE, **extra}
    vals = []
    for i, w in enumerate(words):
        if w in lex:
            v = lex[w]
            if any(t in _NEGATORS for t in words[max(0, i - 2):i]):
                v = -v                            # "not safe" -> threatening
            vals.append(v)
    valence = round(sum(vals) / len(vals), 3) if vals else 0.0
    arousal = round(min(1.0, sum(1 for w in words if w in _AROUSAL) / 3.0), 3)
    return valence, arousal


def _concept(phrase: str, pos: str) -> str:
    """The concept nearest the connective: trailing tokens of a left phrase, leading
    tokens of a right phrase (drops stopwords; keeps up to 3 tokens)."""
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", phrase)
             if w.lower() not in _STOP and len(w) > 1]
    if not words:
        return ""
    words = words[-3:] if pos == "left" else words[:3]
    return " ".join(words).lower()


def causal_links(text: str) -> list[tuple[str, str, int]]:
    """Extract (cause, effect, polarity) triples from causal language."""
    text = _as_text(text)
    out, seen = [], set()
    for rx, pol, rev in _CAUSAL:
        for left, right in rx.findall(text):
            if rev:                                # 'effect <conn> cause'  -> normalize
                cause, effect = _concept(right, "right"), _concept(left, "left")
            else:                                  # 'cause <conn> effect'
                cause, effect = _concept(left, "left"), _concept(right, "right")
            if cause and effect and cause != effect and (cause, effect) not in seen:
                seen.add((cause, effect))
                out.append((cause, effect, pol))
    return out


def salient(text: str, k: int = 5) -> list[str]:
    text = _as_text(text)
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower()) if w not in _STOP and len(w) > 2]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def extract(text: str, lexicon: dict | None = None) -> ContextFrame:
    """Parse an utterance into a ContextFrame. `lexicon` optionally supplies learned
    word-valences (from humind.learning) so affect reflects experience."""
    text = _as_text(text)
    v, a = affect(text, lexicon)
    return ContextFrame(text=text.strip(), intent=classify_intent(text),
                        entities=entities(text), salient=salient(text), valence=v, arousal=a,
                        causes=causal_links(text))
