"""Run every humind demo scenario end to end.

    python demos/run_all.py

Each scenario is independent and builds its own fresh Mind / memory from the
bundled inline transcript, so they can be run in any order or on their own.
Exits 0 on success, so this doubles as a smoke test.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCENARIOS = [
    "01_perception_pipeline",
    "02_reinforcement_learning",
    "03_memory_and_association",
    "04_systems_thinking",
    "05_two_minds_tandem",
]


def main() -> int:
    for name in SCENARIOS:
        mod = importlib.import_module(name)
        mod.main()
    print("\n" + "=" * 70)
    print("  All humind demo scenarios completed.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
