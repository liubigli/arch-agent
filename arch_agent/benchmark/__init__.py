"""Benchmark/evaluation code for the scene agent — not part of the runtime chat path.

Deliberately no re-exports here: `agent.py` imports from `.benchmark.reference_answers`
and `harness.py`/`grounding_checks.py` import back from `..agent`, so re-exporting
submodules at package level would create a circular import.
"""