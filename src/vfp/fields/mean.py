"""Steady mean scalar field from a physical source.

The source is specified as a volumetric emission rate in m3/s and converted here
to a per-cell rate. Cummins instead specify J0 = 1680 ppm/s directly (audit 1.2),
which is a per-cell quantity: the same physical emission spread over a larger cell
gives a smaller ppm/s. Since they also never report dx (audit 1.1), the source
strength in their model is undefined -- two unknowns that only make sense together,
and neither is given.

The practical consequence is that a mesh-convergence study is impossible under
their formulation: refining the grid changes the emitted mass, so successive grids
solve different problems. Depositing a physical amount with mass-conserving
bilinear weights fixes that, and is what makes the M7 convergence index meaningful.

This module is deliberately agnostic about what is being transported. Metofluthrin
from a controlled-release device is another SourceSpec with a different rate and
diffusivity; it needs no code here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid2D
from .abl import Flow
from .operator import BCSpec, LinearFieldSolver

ML_PER_MIN_TO_M3_PER_S = 1e-6 / 60.0


@dataclass(frozen=True)
class SourceSpec:
    """Point sources in physical units."""

    positions_m: np.ndarray  # (M, 2)
    rate_m3_s: np.ndarray  # (M,) volumetric emission of pure tracer

    def __post_init__(self) -> None:
        pos = np.atleast_2d(np.asarray(self.positions_m, dtype=float))
        rate = np.atleast_1d(np.asarray(self.rate_m3_s, dtype=float))
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("SourceSpec.positions_m must have shape (M, 2)")
        if rate.shape != (pos.shape[0],):
            raise ValueError("SourceSpec needs one rate per position")
        if np.any(rate < 0):
            raise ValueError("SourceSpec.rate_m3_s must be >= 0")
        object.__setattr__(self, "positions_m", pos)
        object.__setattr__(self, "rate_m3_s", rate)

    @classmethod
    def hosts(cls, positions_m: np.ndarray, emission_ml_min: float) -> "SourceSpec":
        pos = np.atleast_2d(np.asarray(positions_m, dtype=float))
        return cls(pos, np.full(pos.shape[0], emission_ml_min * ML_PER_MIN_TO_M3_PER_S))

    @property
    def total_m3_s(self) -> float:
        return float(np.sum(self.rate_m3_s))


def source_field(grid: Grid2D, source: SourceSpec, mixing_depth_m: float) -> np.ndarray:
    """Per-cell source in ppm/s.

    A source emitting Q m3/s of pure tracer into a cell of volume dx*dy*h raises
    the volume fraction at Q/(dx*dy*h) per second, hence the 1e6 for ppm. The
    bilinear deposition is the adjoint of interpolation, so the emitted total is
    conserved exactly at any dx -- which is the whole point.
    """
    cell_volume = grid.cell_area * mixing_depth_m
    amounts = 1e6 * source.rate_m3_s / cell_volume
    return grid.deposit(source.positions_m, amounts)


def emitted_ppm_m2_per_s(source: SourceSpec, mixing_depth_m: float) -> float:
    """Total 2D source strength, the quantity a mass balance must reproduce."""
    return 1e6 * source.total_m3_s / mixing_depth_m


def make_solver(grid: Grid2D, flow: Flow, cfg) -> LinearFieldSolver:
    return LinearFieldSolver(
        grid,
        flow.u,
        flow.v,
        flow.k_t,
        reaction=0.0,
        bc=BCSpec.from_config(cfg),
        max_grid_peclet=cfg.numerics.max_grid_peclet,
        scheme=cfg.numerics.advection_scheme,
    )


def solve(grid: Grid2D, flow: Flow, cfg, source: SourceSpec,
          solver: LinearFieldSolver | None = None) -> np.ndarray:
    """Mean concentration in ppm, including any uniform background."""
    solver = solver or make_solver(grid, flow, cfg)
    field = solver.solve(source_field(grid, source, cfg.domain.mixing_depth_m))
    return field + cfg.co2.background_ppm


def analytic_point_source(
    grid: Grid2D, x0: float, y0: float, strength: float, speed: float, k_diff: float
) -> np.ndarray:
    """Infinite-domain steady solution for a point source in uniform +x flow.

        C = (Q / (2 pi K)) exp(U xi / (2K)) K0(U r / (2K))

    with xi measured downstream. Used only to verify the discretisation.
    """
    from scipy.special import k0e

    x, y = grid.mesh()
    xi = x - x0
    r = np.hypot(xi, y - y0)
    r = np.maximum(r, 1e-12)
    # k0e(z) = exp(z) K0(z), so exp(U xi/2K) K0(U r/2K) = k0e(z) exp(U(xi - r)/2K),
    # and xi - r <= 0 always: no overflow anywhere in the domain.
    z = speed * r / (2.0 * k_diff)
    return (strength / (2.0 * np.pi * k_diff)) * k0e(z) * np.exp(speed * (xi - r) / (2.0 * k_diff))
