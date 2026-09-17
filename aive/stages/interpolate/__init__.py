"""Frame interpolation stage (M2).

RIFE 4.25 family with TAS-style gap planning, de-duplication and scene-cut
protection by construction: the orchestrator runs this stage once per scene
segment, so interpolation never crosses a cut.
"""
