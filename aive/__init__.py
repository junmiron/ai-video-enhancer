"""AI Video Enhancer (aive).

Pipeline: ingest -> scene split -> dedup -> interpolate -> upscale -> artifact QA -> encode.
M0/M1 scope: core pipeline, FFmpeg I/O, checkpointing, model manager, inference backend layer.
"""

__version__ = "0.1.0"
