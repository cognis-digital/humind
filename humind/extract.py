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
_STOP = set("a an the of to in on at is are was were be been and or but for with this that "
            "it its as by from we i you he she they them our your".split())
# all-caps status/severity markers that read as entities but aren't named things
_NONENTITY = set("critical urgent alert warning notice note high low medium fyi update status "
                 "emergency caution danger info report flash priority routine".split())


@dataclass
class ContextFrame:
    text: str
    intent: str                              # inform | request | query | propose | agree | refuse
    entities: list[str] = field(default_factory=list)
    salient: list[str] = field(default_factory=list)
    valence: float = 0.0                     # negative = threatening/bad, positive = good
    arousal: float = 0.0                     # 0..1 urgency
    notes: str = ""                          # optional LLM-enrichment annotation (see addins)

    def to_dict(self) -> dict:
        return asdict(self)


def classify_intent(text: str) -> str:
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


def affect(text: str) -> tuple[float, float]:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0, 0.0
    vals = [_VALENCE[w] for w in words if w in _VALENCE]
    valence = round(sum(vals) / len(vals), 3) if vals else 0.0
    arousal = round(min(1.0, sum(1 for w in words if w in _AROUSAL) / 3.0), 3)
    return valence, arousal


def salient(text: str, k: int = 5) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9-]+", text.lower()) if w not in _STOP and len(w) > 2]
    freq: dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def extract(text: str) -> ContextFrame:
    v, a = affect(text)
    return ContextFrame(text=text.strip(), intent=classify_intent(text),
                        entities=entities(text), salient=salient(text), valence=v, arousal=a)
