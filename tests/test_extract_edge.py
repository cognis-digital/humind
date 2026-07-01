"""Edge cases and error paths for the perception stage (humind.extract).

Covers empty/whitespace/unicode input, intent-precedence ordering, negation
windows and boundaries, entity de-duplication and non-entity filtering, numeric
IDs, salience ranking and short-word filtering, causal reversed connectives and
polarity, and the type-validation error paths added in v0.3.1.
"""

from __future__ import annotations

import pytest

from humind import extract
from humind.extract import (affect, causal_links, classify_intent, entities,
                            salient, ContextFrame, _as_text)


# --- type validation / error paths -------------------------------------------
@pytest.mark.parametrize("fn", [extract, classify_intent, entities, affect,
                                causal_links, salient])
@pytest.mark.parametrize("bad", [None, 123, 4.5, ["a"], {"x": 1}, b"bytes"])
def test_non_string_input_raises_typeerror(fn, bad):
    with pytest.raises(TypeError):
        fn(bad)


def test_as_text_names_the_argument():
    with pytest.raises(TypeError) as ei:
        _as_text(None, arg="utterance")
    assert "utterance" in str(ei.value)


def test_salient_negative_k_raises():
    with pytest.raises(ValueError):
        salient("some text here", k=-1)


# --- empty / whitespace / unicode --------------------------------------------
def test_empty_string_is_neutral_inform():
    f = extract("")
    assert f.intent == "inform"
    assert f.entities == [] and f.salient == []
    assert f.valence == 0.0 and f.arousal == 0.0 and f.causes == []


def test_whitespace_only_is_neutral():
    f = extract("   \t\n  ")
    assert f.text == "" and f.intent == "inform" and f.valence == 0.0


def test_unicode_does_not_crash_and_extracts():
    f = extract("URGENT: vessel Ñandú went dark — risk élevé")
    assert isinstance(f, ContextFrame)
    assert f.arousal > 0                    # 'urgent' still counts


def test_text_is_stripped():
    assert extract("   hello world   ").text == "hello world"


# --- intent precedence -------------------------------------------------------
def test_query_beats_everything_when_question_mark():
    # a question that also contains a refusal word is still a query (? checked first)
    assert classify_intent("No, is that safe?") == "query"


def test_refuse_beats_agree_and_request():
    assert classify_intent("No, please stop") == "refuse"


def test_agree_beats_request():
    assert classify_intent("Ok, could you proceed") == "agree"


def test_propose_is_lowest_priority_cue():
    assert classify_intent("Maybe reroute") == "propose"


def test_plain_statement_is_inform():
    assert classify_intent("The vessel is at anchor") == "inform"


# --- affect: negation windows and boundaries --------------------------------
def test_negation_within_two_tokens_flips():
    assert affect("this is not safe")[0] < 0
    assert affect("this is safe")[0] > 0


def test_negation_beyond_window_does_not_flip():
    # 'not' is 3 tokens before 'safe' -> outside the 2-token window, stays positive
    v = affect("not really quite very safe")[0]
    assert v > 0


def test_double_negation_context_still_flips_once():
    # single negator in window flips sign; we only assert the sign changed
    assert affect("never safe")[0] < 0


def test_arousal_saturates_at_one():
    # many arousal words -> arousal capped at 1.0
    a = affect("urgent critical danger threat alarm emergency now immediately")[1]
    assert a == 1.0


def test_affect_with_no_lexicon_words_is_zero():
    assert affect("the quick brown fox jumped")[0] == 0.0


def test_learned_lexicon_overrides_merge():
    v_base = affect("a widget appeared")[0]
    v_learned = affect("a widget appeared", {"widget": -0.9})[0]
    assert v_base == 0.0 and v_learned < 0


def test_learned_word_also_negation_flips():
    assert affect("no widget", {"widget": -0.8})[0] > 0     # neg flips learned neg -> pos


# --- entities ----------------------------------------------------------------
def test_entities_dedupe_preserving_order():
    ents = entities("NEPTUNE-STAR met NEPTUNE-STAR near GHOST-RUNNER")
    assert ents == ["NEPTUNE-STAR", "GHOST-RUNNER"]


def test_nonentity_severity_words_filtered():
    # CRITICAL / URGENT / ALERT read like entities but are status markers
    assert "CRITICAL" not in entities("CRITICAL Situation Report")
    assert "URGENT" not in entities("URGENT dispatch")


def test_numeric_ids_are_entities():
    ents = entities("track contact IMO-9321483 now")
    assert any("9321483" in e for e in ents)


def test_lowercase_prose_has_no_capitalised_entities():
    assert entities("the vessel went dark near the strait") == []


# --- salience ----------------------------------------------------------------
def test_salience_ranks_by_frequency():
    s = salient("drone drone radar drone acoustic radar")
    assert s[0] == "drone" and "radar" in s


def test_salience_drops_short_and_stopwords():
    # 'is'/'on'/'it' are <=2 chars or stopwords -> excluded
    assert salient("it is on") == []


def test_salience_respects_k():
    s = salient("alpha beta gamma delta epsilon zeta", k=2)
    assert len(s) == 2


def test_salience_k_zero_returns_empty():
    assert salient("alpha beta gamma", k=0) == []


# --- causal links ------------------------------------------------------------
def test_causal_positive_polarity():
    links = causal_links("darkness leads to escalation")
    assert any(p == 1 for _, _, p in links)


def test_causal_negative_polarity():
    links = causal_links("patrols reduces smuggling")
    assert any(c == "patrols" and p == -1 for c, e, p in links)


def test_causal_reversed_connective_normalises_direction():
    links = causal_links("escalation because of the gap")
    # 'because of' -> effect <conn> cause, normalised so cause precedes effect
    assert any(c == "gap" and e.startswith("escalation") for c, e, p in links)


def test_causal_ignores_self_reference():
    # a statement where both sides collapse to the same concept yields nothing
    assert causal_links("risk drives risk") == []


def test_causal_dedupes_repeated_triples():
    txt = "gap leads to risk. gap leads to risk."
    links = causal_links(txt)
    assert len(links) == len({(c, e) for c, e, _ in links})


def test_extract_frame_to_dict_roundtrips_fields():
    d = extract("URGENT: NEPTUNE-STAR is a high risk threat").to_dict()
    for key in ("text", "intent", "entities", "salient", "valence",
                "arousal", "causes", "surprise", "notes"):
        assert key in d
