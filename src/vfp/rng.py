"""Reproducible random streams.

One Generator per named stream per run, addressed by (field, replicate, stream)
through a SeedSequence spawn key. Indices are addressed directly rather than by
materialising a spawn list, so `drivers/sweep.py` can jump to realization 37
without generating the first 36.

Deliberate consequence, tested in `tests/test_reproducibility.py`: streams are
per-run, not per-agent, so changing `agents.n` changes every agent's draws. N is
therefore part of a run's identity. Per-agent streams would fix that at a
10-100x cost in stepping time, which is not worth it.
"""

from __future__ import annotations

import numpy as np

STREAM_NAMES: tuple[str, ...] = (
    "field",       # wind realization: direction, u* jitter, meander phase
    "agent_init",  # release positions and initial headings
    "turbulence",  # AR(1) latent driving the concentration PDF
    "heading",     # von Mises heading draws
    "wind_ou",     # Ornstein-Uhlenbeck velocity perturbation
    "crosswind",   # crosswind leg durations and sign flips
    "knockdown",   # reserved for the repellent phase
)


def streams(
    master_seed: int, field_idx: int = 0, replicate_idx: int = 0
) -> dict[str, np.random.Generator]:
    return {
        name: np.random.default_rng(
            np.random.SeedSequence(master_seed, spawn_key=(field_idx, replicate_idx, i))
        )
        for i, name in enumerate(STREAM_NAMES)
    }
