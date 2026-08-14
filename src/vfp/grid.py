"""Uniform cell-centred Cartesian grid.

Cell (ix, iy) is centred at ((ix + 1/2) dx, (iy + 1/2) dy) and ravels to
k = iy * nx + ix, so the five-point stencil neighbours are k +/- 1 and k +/- nx.

Interpolation weights are computed once and applied to many fields. The agent
step reads the mean and the variance at five probe points each, so sharing the
weight computation across fields is worth the small amount of API surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InterpWeights:
    """Bilinear stencil for a batch of query points, reusable across fields."""

    k00: np.ndarray
    k10: np.ndarray
    k01: np.ndarray
    k11: np.ndarray
    w00: np.ndarray
    w10: np.ndarray
    w01: np.ndarray
    w11: np.ndarray


class Grid2D:
    def __init__(self, lx: float, ly: float, dx: float) -> None:
        nx = int(round(lx / dx))
        ny = int(round(ly / dx))
        if nx < 2 or ny < 2:
            raise ValueError(f"grid must be at least 2x2 cells, got {nx}x{ny}")
        self.lx = float(lx)
        self.ly = float(ly)
        self.dx = float(dx)
        self.dy = float(dx)
        self.nx = nx
        self.ny = ny
        self.n = nx * ny
        self.xc = (np.arange(nx) + 0.5) * self.dx
        self.yc = (np.arange(ny) + 0.5) * self.dy
        self.cell_area = self.dx * self.dy

    def __repr__(self) -> str:
        return f"Grid2D({self.lx}x{self.ly} m, dx={self.dx} m, {self.nx}x{self.ny} cells)"

    @classmethod
    def from_config(cls, cfg) -> "Grid2D":
        return cls(cfg.domain.lx_m, cfg.domain.ly_m, cfg.numerics.dx_m)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """(X, Y) with shape (ny, nx), matching field layout."""
        return np.meshgrid(self.xc, self.yc)

    def index(self, ix: np.ndarray | int, iy: np.ndarray | int) -> np.ndarray | int:
        return iy * self.nx + ix

    def zeros(self) -> np.ndarray:
        return np.zeros((self.ny, self.nx))

    def interp_weights(self, x: np.ndarray, y: np.ndarray) -> InterpWeights:
        """Bilinear weights, clamped at the boundary.

        Query points outside the cell-centre hull read the boundary value rather
        than extrapolating. Agents are allowed to leave the CO2 region while the
        flight domain continues, so this path is taken routinely, not rarely.
        """
        fx = np.clip(x / self.dx - 0.5, 0.0, self.nx - 1.0)
        fy = np.clip(y / self.dy - 0.5, 0.0, self.ny - 1.0)
        i0 = np.minimum(np.floor(fx).astype(np.intp), self.nx - 2)
        j0 = np.minimum(np.floor(fy).astype(np.intp), self.ny - 2)
        tx = fx - i0
        ty = fy - j0
        k00 = j0 * self.nx + i0
        return InterpWeights(
            k00=k00,
            k10=k00 + 1,
            k01=k00 + self.nx,
            k11=k00 + self.nx + 1,
            w00=(1.0 - tx) * (1.0 - ty),
            w10=tx * (1.0 - ty),
            w01=(1.0 - tx) * ty,
            w11=tx * ty,
        )

    @staticmethod
    def apply(field: np.ndarray, w: InterpWeights) -> np.ndarray:
        flat = field.reshape(-1)
        return (
            flat[w.k00] * w.w00
            + flat[w.k10] * w.w10
            + flat[w.k01] * w.w01
            + flat[w.k11] * w.w11
        )

    def sample(self, field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.apply(field, self.interp_weights(x, y))

    def gradient(self, field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(df/dx, df/dy), second order interior, second order one-sided at edges."""
        dfdy, dfdx = np.gradient(field, self.dy, self.dx, edge_order=2)
        return dfdx, dfdy

    def deposit(self, positions: np.ndarray, amounts: np.ndarray) -> np.ndarray:
        """Bilinear (area-weighted) deposition of point amounts onto cells.

        The adjoint of `apply`, so the total deposited equals sum(amounts) exactly
        regardless of dx. This is what makes the mesh-convergence study meaningful:
        under Cummins' per-cell J0 in ppm/s, refining the grid silently changes the
        emission rate, so a converged field would be a field of a different problem.
        """
        x, y = np.asarray(positions)[:, 0], np.asarray(positions)[:, 1]
        if np.any((x < 0) | (x > self.lx) | (y < 0) | (y > self.ly)):
            raise ValueError("deposit: source positions must lie inside the domain")
        w = self.interp_weights(x, y)
        out = np.zeros(self.n)
        amounts = np.asarray(amounts, dtype=float)
        for k, weight in ((w.k00, w.w00), (w.k10, w.w10), (w.k01, w.w01), (w.k11, w.w11)):
            np.add.at(out, k, amounts * weight)
        return out.reshape(self.ny, self.nx)
