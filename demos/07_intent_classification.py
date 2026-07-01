"""Scenario 7 - speech acts (dialogue-systems builders).

Every utterance carries an **intent** (a speech act): is the speaker informing,
asking, requesting, proposing, agreeing, or refusing? humind classifies it from
transparent surface cues with a fixed precedence — a trailing '?' wins first, then
refuse / agree / request / propose, and anything else is 'inform'. This demo runs
a varied mini-dialogue and then shows the precedence order on deliberately
ambiguous lines (a refusal inside a question, an 'ok' inside a request).
"""
from _common import extract, rule


DIALOGUE = [
    "The convoy is holding position.",
    "Is GHOST-RUNNER inside the exclusion zone?",
    "Please reroute away from the corridor.",
    "I think we should slow to ten knots.",
    "Agreed, that plan is confirmed.",
    "No, we cannot divert the escort.",
]

AMBIGUOUS = [
    ("No, is that even safe?", "trailing '?' -> query wins over the 'no'"),
    ("Ok, could you confirm?", "trailing '?' -> query wins over agree/request"),
    ("No, please stop.", "refuse checked before request"),
    ("Ok, could you proceed.", "agree checked before request"),
]


def main() -> None:
    rule("INTENT / SPEECH ACTS  -  what is the speaker DOING with this utterance")
    print("\nA mini-dialogue, one intent per line:\n")
    for line in DIALOGUE:
        print(f"   intent={extract(line).intent:<8}  \"{line}\"")

    print("\nPrecedence on ambiguous lines (query > refuse > agree > request > propose):\n")
    for line, why in AMBIGUOUS:
        print(f"   intent={extract(line).intent:<8}  \"{line}\"")
        print(f"            -> {why}")

    print("\nIntent lines up 1:1 with an agentlex performative, so a classified")
    print("utterance can be spoken to another agent without re-interpretation.")


if __name__ == "__main__":
    main()
