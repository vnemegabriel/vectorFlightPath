"""Steady advection-diffusion-reaction operator on a cell-centred finite volume grid.

Solves  div(V phi) - div(K grad phi) + r phi = S  as a sparse linear system rather
than by marching in time. Three reasons, in order of weight:

1. It removes dt from the field entirely. Cummins never report their timestep, and
   with an unreported dt their field is not reproducible (audit 1.1). Here only the
   agent timestep remains, and that is convergence-tested on its own.
2. It makes the isoprotection sweep affordable. The operator depends on the flow,
   not on the source, so one splu() per wind realization serves every host position
   in the sweep at the cost of a back-substitution each.
3. The boundary conditions can be imposed properly. Cummins used zero normal
   gradient on every face including the inflow, where Dirichlet C = 0 is correct
   (audit 2.4); Neumann at an inflow face permits spurious accumulation.

Advection uses central differences. That is not a soft preference: with the
physically correct eddy diffusivity K_t ~ 1.04e-2 m2/s and dx = 0.025 m, the grid
Peclet number is 0.48, so central differencing is non-oscillatory AND has exactly
zero numerical diffusivity. Cummins used first-order upwind, whose numerical
diffusivity U dx / 2 lands between 88x and 1100x molecular -- the same order as the
turbulent diffusivity it was standing in for. Their plume width was therefore set
by a discretisation artefact. `assemble` refuses to run central above
`max_grid_peclet` rather than degrading quietly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..grid import Grid2D

INFLOW, OUTFLOW, LATERAL = 0, 1, 2


class OperatorError(ValueError):
    pass


@dataclass(frozen=True)
class BCSpec:
    inflow: str = "dirichlet"
    outflow: str = "advective"
    lateral: str = "neumann"

    @classmethod
    def from_config(cls, cfg) -> "BCSpec":
        b = cfg.boundary
        return cls(inflow=b.scalar_inflow, outflow=b.scalar_outflow, lateral=b.scalar_lateral)

    def kind_for(self, role: np.ndarray) -> np.ndarray:
        """Map INFLOW/OUTFLOW/LATERAL codes to the configured treatment names."""
        out = np.empty(role.shape, dtype="<U10")
        out[role == INFLOW] = self.inflow
        out[role == OUTFLOW] = self.outflow
        out[role == LATERAL] = self.lateral
        return out


def _face_values(cell: np.ndarray, axis: int) -> np.ndarray:
    """Arithmetic face averages, with boundary faces taking the adjacent cell value."""
    if axis == 1:
        return np.concatenate(
            [cell[:, :1], 0.5 * (cell[:, :-1] + cell[:, 1:]), cell[:, -1:]], axis=1
        )
    return np.concatenate([cell[:1, :], 0.5 * (cell[:-1, :] + cell[1:, :]), cell[-1:, :]], axis=0)


def _classify(normal_in: np.ndarray, tol: float) -> np.ndarray:
    """Inflow / outflow / lateral from the inward normal velocity at a boundary face."""
    role = np.full(normal_in.shape, LATERAL, dtype=np.int8)
    role[normal_in > tol] = INFLOW
    role[normal_in < -tol] = OUTFLOW
    return role


def assemble(
    grid: Grid2D,
    u: np.ndarray,
    v: np.ndarray,
    k_diff: np.ndarray,
    reaction: np.ndarray | float = 0.0,
    bc: BCSpec | None = None,
    max_grid_peclet: float = 2.0,
    scheme: str = "central",
) -> sp.csc_matrix:
    bc = bc or BCSpec()
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy

    speed = np.hypot(u, v)
    if scheme == "central":
        pe = float(np.max(speed * dx / k_diff))
        if pe >= max_grid_peclet:
            dx_needed = max_grid_peclet * float(np.min(k_diff)) / float(np.max(speed))
            raise OperatorError(
                f"grid Peclet number {pe:.2f} >= {max_grid_peclet}: central differencing "
                f"would oscillate. Refine dx below {dx_needed:.4g} m, or set "
                f"numerics.advection_scheme = 'upwind' and accept a numerical "
                f"diffusivity of U*dx/2 = {float(np.max(speed)) * dx / 2:.3g} m2/s."
            )
    elif scheme != "upwind":
        raise OperatorError(f"unknown advection scheme {scheme!r}")

    uf = _face_values(u, axis=1)  # (ny, nx+1)
    vf = _face_values(v, axis=0)  # (ny+1, nx)
    kfx = _face_values(k_diff, axis=1)
    kfy = _face_values(k_diff, axis=0)

    aP = np.zeros((ny, nx)) + np.asarray(reaction, dtype=float)
    aE = np.zeros((ny, nx))
    aW = np.zeros((ny, nx))
    aN = np.zeros((ny, nx))
    aS = np.zeros((ny, nx))

    def _upwind_split(vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Face value weights (w_left, w_right) for the donor cell."""
        if scheme == "central":
            return 0.5 * np.ones_like(vel), 0.5 * np.ones_like(vel)
        pos = (vel > 0).astype(float)
        return pos, 1.0 - pos

    # --- interior x-faces: between cell (i-1, j) [left] and (i, j) [right] ------
    ufi = uf[:, 1:nx]
    kfi = kfx[:, 1:nx]
    wl, wr = _upwind_split(ufi)
    aP[:, :-1] += ufi * wl / dx + kfi / dx**2
    aE[:, :-1] += ufi * wr / dx - kfi / dx**2
    aP[:, 1:] += -ufi * wr / dx + kfi / dx**2
    aW[:, 1:] += -ufi * wl / dx - kfi / dx**2

    # --- interior y-faces ------------------------------------------------------
    vfi = vf[1:ny, :]
    kfj = kfy[1:ny, :]
    wl, wr = _upwind_split(vfi)
    aP[:-1, :] += vfi * wl / dy + kfj / dy**2
    aN[:-1, :] += vfi * wr / dy - kfj / dy**2
    aP[1:, :] += -vfi * wr / dy + kfj / dy**2
    aS[1:, :] += -vfi * wl / dy - kfj / dy**2

    # --- boundary faces -------------------------------------------------------
    tol = 1e-9 * float(np.max(speed))

    def _apply_boundary(vel_face, k_face, normal_in, sign, spacing, target_slice):
        """sign = +1 where the outward normal is +axis (east/north), -1 otherwise."""
        kind = bc.kind_for(_classify(normal_in, tol))
        add = np.zeros_like(vel_face)
        is_dirichlet = kind == "dirichlet"
        is_advective = kind == "advective"
        add[is_dirichlet] = 2.0 * k_face[is_dirichlet] / spacing**2
        add[is_advective] = sign * vel_face[is_advective] / spacing
        aP[target_slice] += add
        return bool(np.any(is_dirichlet | is_advective))

    has_outlet = False
    has_outlet |= _apply_boundary(uf[:, 0], kfx[:, 0], uf[:, 0], -1.0, dx, (slice(None), 0))
    has_outlet |= _apply_boundary(uf[:, nx], kfx[:, nx], -uf[:, nx], +1.0, dx, (slice(None), nx - 1))
    has_outlet |= _apply_boundary(vf[0, :], kfy[0, :], vf[0, :], -1.0, dy, (0, slice(None)))
    has_outlet |= _apply_boundary(vf[ny, :], kfy[ny, :], -vf[ny, :], +1.0, dy, (ny - 1, slice(None)))

    if not has_outlet and not np.any(np.asarray(reaction, dtype=float) > 0.0):
        raise OperatorError(
            "no boundary face lets mass leave and there is no sink, so the steady "
            "problem is singular: a source would accumulate without bound. This is "
            "the windless configuration; set boundary.scalar_lateral = 'dirichlet' "
            "so the domain is embedded in clean air."
        )

    # --- assemble -------------------------------------------------------------
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    k = (jj * nx + ii).ravel()
    rows = [k]
    cols = [k]
    data = [aP.ravel()]
    for coeff, mask, offset in (
        (aE, ii < nx - 1, 1),
        (aW, ii > 0, -1),
        (aN, jj < ny - 1, nx),
        (aS, jj > 0, -nx),
    ):
        m = mask.ravel()
        rows.append(k[m])
        cols.append(k[m] + offset)
        data.append(coeff.ravel()[m])

    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(grid.n, grid.n),
    ).tocsc()


def boundary_efflux(
    grid: Grid2D, u: np.ndarray, v: np.ndarray, k_diff: np.ndarray,
    phi: np.ndarray, bc: BCSpec | None = None
) -> float:
    """Net scalar flux out through the domain boundary, computed independently.

    Checking this against the total source is a genuine test: it verifies that the
    boundary treatment is consistent with the interior stencil, which nothing in
    the linear solve itself would reveal.
    """
    bc = bc or BCSpec()
    nx, ny, dx, dy = grid.nx, grid.ny, grid.dx, grid.dy
    uf = _face_values(u, axis=1)
    vf = _face_values(v, axis=0)
    kfx = _face_values(k_diff, axis=1)
    kfy = _face_values(k_diff, axis=0)
    tol = 1e-9 * float(np.max(np.hypot(u, v)))

    total = 0.0
    for vel_face, k_face, normal_in, phi_edge, spacing, width in (
        (uf[:, 0], kfx[:, 0], uf[:, 0], phi[:, 0], dx, dy),
        (-uf[:, nx], kfx[:, nx], -uf[:, nx], phi[:, nx - 1], dx, dy),
        (vf[0, :], kfy[0, :], vf[0, :], phi[0, :], dy, dx),
        (-vf[ny, :], kfy[ny, :], -vf[ny, :], phi[ny - 1, :], dy, dx),
    ):
        kind = bc.kind_for(_classify(normal_in, tol))
        # vel_face here is already the INWARD normal velocity component.
        out = np.zeros_like(phi_edge)
        dirichlet = kind == "dirichlet"
        advective = kind == "advective"
        out[dirichlet] = 2.0 * k_face[dirichlet] * phi_edge[dirichlet] / spacing
        out[advective] = -vel_face[advective] * phi_edge[advective]
        total += float(np.sum(out) * width)
    return total


class LinearFieldSolver:
    """One factorisation, many right-hand sides.

    The isoprotection sweep moves the host and keeps the flow, so this is the
    difference between 8 factorisations and 960 full solves.
    """

    def __init__(
        self,
        grid: Grid2D,
        u: np.ndarray,
        v: np.ndarray,
        k_diff: np.ndarray,
        reaction: np.ndarray | float = 0.0,
        bc: BCSpec | None = None,
        max_grid_peclet: float = 2.0,
        scheme: str = "central",
    ) -> None:
        self.grid = grid
        self.matrix = assemble(grid, u, v, k_diff, reaction, bc, max_grid_peclet, scheme)
        self._lu = spla.splu(self.matrix)

    def solve(self, source: np.ndarray) -> np.ndarray:
        phi = self._lu.solve(np.asarray(source, dtype=float).reshape(-1))
        return phi.reshape(self.grid.ny, self.grid.nx)

    def residual(self, phi: np.ndarray, source: np.ndarray) -> float:
        r = self.matrix @ phi.reshape(-1) - np.asarray(source, dtype=float).reshape(-1)
        denom = np.linalg.norm(np.asarray(source, dtype=float).reshape(-1))
        return float(np.linalg.norm(r) / denom) if denom > 0 else float(np.linalg.norm(r))
