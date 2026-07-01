"""humind — a cognitive-architecture-inspired NL context engine.

Extracts human-meaningful context from natural language (entities, intent, affect,
salience) and holds it in working / episodic / semantic memory — then speaks to other
agents in `agentlex`. It models *facets* of cognition (memory, attention, affect,
intent); it does not literally replicate the brain.
"""

from humind.extract import (ContextFrame, affect, causal_links, classify_intent,
                            entities, extract, salient)
from humind.learning import LexiconLearner, ValueLearner
from humind.memory import (AssociativeMemory, CognitiveMemory, EpisodicMemory,
                           SemanticMemory, WorkingMemory)
from humind.mind import Mind
from humind.systems import CausalGraph

__version__ = "0.3.2"

__all__ = [
    "extract", "ContextFrame", "classify_intent", "entities", "affect", "salient",
    "causal_links", "WorkingMemory", "EpisodicMemory", "SemanticMemory",
    "AssociativeMemory", "CognitiveMemory", "ValueLearner", "LexiconLearner",
    "CausalGraph", "Mind", "__version__",
]
