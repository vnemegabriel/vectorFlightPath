"""Steady scalar-variance transport.

    U . grad(c'2) = div[(K_t/sigma_c) grad(c'2)] + 2 K_t |grad C|^2 - 2 C_d (eps/k) c'2

Production comes from the mean gradient, destruction from scalar dissipation on
the mechanical timescale k/eps. Solving this is what lets an agent read an
intermittent signal without storing a time series: the pair (mean, variance) is
enough to parameterise a local concentration PDF, which `sampler.py` then draws
from.

Why it matters here specifically. Cummins' own literature review states that
sustained upwind flight occurs in response to an intermittent CO2 signal and not
to a uniform one, and that a wide well-mixed plume can fail to induce the flight
that a turbulent plume of the same mean concentration does induce. Their model
then represents neither: the agent reads a smoothed field, so the concentration
statistics it experiences have the wrong PDF for a responder whose response is
non-linear with a threshold (audit 3.2). The variance field is what closes that
gap.

Same assembler as the mean, with the dissipation term on the diagonal and the
production term as the right-hand side. One factorisation per wind realization,
reused across every host position in an isoprotection sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid2D
from .abl import Flow
from .operator import BCSpec, LinearFieldSolver

# Instantaneous concentration cannot exceed that of the undiluted tracer.
PURE_TRACER_PPM = 1e6


@dataclass(frozen=True)
class VarianceResult:
    field: np.ndarray  # c'2, ppm^2
    clipped_fraction: float  # within the plume mask only
    intensity_p50: float
    intensity_p99: float

    def report(self) -> str:
        return (
            f"variance: clipped {100 * self.clipped_fraction:.2f}% of plume cells, "
            f"fluctuation intensity p50={self.intensity_p50:.2f} p99={self.intensity_p99:.2f}"
        )


def dissipation_rate(flow: Flow, c_dissipation: float) -> float:
    """2 C_d eps/k, the reciprocal decay timescale of scalar variance."""
    if flow.tke <= 0.0:
        return 0.0
    return 2.0 * c_dissipation * flow.dissipation / flow.tke


def make_solver(grid: Grid2D, flow: Flow, cfg) -> LinearFieldSolver:
    reaction = dissipation_rate(flow, cfg.wind.c_dissipation)
    bc = BCSpec.from_config(cfg)
    if reaction <= 0.0:
        # Windless: no scalar dissipation, so the variance problem needs an outlet
        # of its own or it is as singular as the mean problem would be.
        bc = BCSpec(bc.inflow, bc.outflow, "dirichlet")
    return LinearFieldSolver(
        grid,
        flow.u,
        flow.v,
        flow.k_t / cfg.wind.sc_variance,
        reaction=reaction,
        bc=bc,
        max_grid_peclet=cfg.numerics.max_grid_peclet,
        scheme=cfg.numerics.advection_scheme,
    )


def production(grid: Grid2D, flow: Flow, mean_field: np.ndarray) -> np.ndarray:
    dcdx, dcdy = grid.gradient(mean_field)
    return 2.0 * flow.k_t * (dcdx**2 + dcdy**2)


def solve(
    grid: Grid2D,
    flow: Flow,
    cfg,
    mean_field: np.ndarray,
    solver: LinearFieldSolver | None = None,
    plume_threshold_ppm: float | None = None,
) -> VarianceResult:
    solver = solver or make_solver(grid, flow, cfg)
    raw = solver.solve(production(grid, flow, mean_field))

    # Realizability: 0 <= c'2 <= C (C_max - C).
    upper = np.maximum(mean_field * (PURE_TRACER_PPM - mean_field), 0.0)
    clipped = np.clip(raw, 0.0, upper)

    # Diagnostics are reported over the sensible plume only. Far from the source
    # both the mean and the variance underflow to numerical noise, and a ratio of
    # two such numbers says nothing about the closure. Reporting it would drown the
    # signal a large clipped fraction is supposed to send -- that the closure
    # constants are wrong for this configuration.
    threshold = (
        plume_threshold_ppm
        if plume_threshold_ppm is not None
        else 1e-3 * float(mean_field.max())
    )
    plume = mean_field > max(threshold, 1e-30)
    if not np.any(plume):
        return VarianceResult(clipped, 0.0, 0.0, 0.0)

    intensity = np.sqrt(clipped[plume]) / mean_field[plume]
    return VarianceResult(
        field=clipped,
        clipped_fraction=float(
            np.mean(~np.isclose(raw[plume], clipped[plume], rtol=1e-9, atol=1e-30))
        ),
        intensity_p50=float(np.percentile(intensity, 50)),
        intensity_p99=float(np.percentile(intensity, 99)),
    )
