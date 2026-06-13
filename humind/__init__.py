"""humind — a cognitive-architecture-inspired NL context engine.

Extracts human-meaningful context from natural language (entities, intent, affect,
salience) and holds it in working / episodic / semantic memory — then speaks to other
agents in `agentlex`. It models *facets* of cognition (memory, attention, affect,
intent); it does not literally replicate the brain.
"""

from humind.extract import ContextFrame, affect, classify_intent, entities, extract, salient
from humind.memory import CognitiveMemory, EpisodicMemory, SemanticMemory, WorkingMemory
from humind.mind import Mind

__version__ = "0.1.0"

__all__ = [
    "extract", "ContextFrame", "classify_intent", "entities", "affect", "salient",
    "WorkingMemory", "EpisodicMemory", "SemanticMemory", "CognitiveMemory", "Mind",
    "__version__",
]
