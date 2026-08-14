"""Atmospheric boundary layer: mean wind and turbulence from similarity theory.

Replaces Cummins' wind model, which was a uniform 0.2 m/s plus a spatially white
Gaussian noise resampled every 2 s. That noise has no declared correlation length
(audit 1.5); being white in space it produces Fickian dispersion with an effective
diffusivity of ~1.1e-2 m2/s and no meander, no filaments, no intermittency. The
authors evidently knew, because they then had to impose a meandering plume by
hand as a closed kinematic formula -- and that imposed geometry produces the
largest quantitative effect in the whole paper.

Here the turbulence is instead specified by (u*, z0, L_MO), from which k, epsilon,
the eddy diffusivity and the integral scales follow by standard relations.

Local-equilibrium closure, for the record. With
    dU/dz = (u*/(kappa z)) phi_m,     eps = (u*^3/(kappa z)) (phi_m - zeta),
and production balancing dissipation under nu_t = C_mu k^2/eps, one gets
    k = u*^2 (phi_m - zeta) / (sqrt(C_mu) phi_m).
In neutral conditions this collapses to the familiar k = u*^2/sqrt(C_mu), and
nu_t = C_mu k^2/eps reduces exactly to kappa u* z. That identity is asserted in
`tests/test_abl.py`, because it is the one place a sign or exponent slip would
otherwise pass unnoticed.

Reference values at z = 1 m, u* = 0.0178 m/s, z0 = 0.01 m, neutral:
    U = 0.200 m/s     k = 1.056e-3 m2/s2    eps = 1.376e-5 m2/s3
    K_t = 1.043e-2 m2/s (= 650x molecular)  u' = 0.0265 m/s
    L_t = 0.410 m     tau_L = 1.81 s
Note K_t is the same order as the numerical diffusivity Cummins' first-order
upwind scheme would have produced. Their plume width was set by a discretisation
artefact that happened to land near the physically correct value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..grid import Grid2D


def _zeta(z: float | np.ndarray, obukhov_length_m: float) -> np.ndarray:
    """Stability parameter z/L. L = 0 encodes neutral, so zeta = 0."""
    z = np.asarray(z, dtype=float)
    if obukhov_length_m == 0.0:
        return np.zeros_like(z)
    return z / obukhov_length_m


def phi_m(zeta: np.ndarray) -> np.ndarray:
    """Businger-Dyer dimensionless wind shear."""
    zeta = np.asarray(zeta, dtype=float)
    return np.where(zeta >= 0.0, 1.0 + 5.0 * zeta, (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** -0.25)


def psi_m(zeta: np.ndarray) -> np.ndarray:
    """Integrated stability correction, continuous at zeta = 0."""
    zeta = np.asarray(zeta, dtype=float)
    x = (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** 0.25
    unstable = (
        2.0 * np.log(0.5 * (1.0 + x))
        + np.log(0.5 * (1.0 + x * x))
        - 2.0 * np.arctan(x)
        + 0.5 * np.pi
    )
    return np.where(zeta >= 0.0, -5.0 * zeta, unstable)


def wind_speed(
    z: float, u_star: float, z0: float, obukhov_length_m: float = 0.0, kappa: float = 0.41
) -> float:
    """Log law with stability correction."""
    zeta_z = _zeta(z, obukhov_length_m)
    zeta_0 = _zeta(z0, obukhov_length_m)
    return float(
        (u_star / kappa) * (np.log(z / z0) - psi_m(zeta_z) + psi_m(zeta_0))
    )


def u_star_from_speed(
    speed: float, z: float, z0: float, obukhov_length_m: float = 0.0, kappa: float = 0.41
) -> float:
    """Invert the log law. Exact, since psi_m depends on z/L and not on u*."""
    zeta_z = _zeta(z, obukhov_length_m)
    zeta_0 = _zeta(z0, obukhov_length_m)
    return float(kappa * speed / (np.log(z / z0) - psi_m(zeta_z) + psi_m(zeta_0)))


def tke(z: float, u_star: float, obukhov_length_m: float = 0.0, c_mu: float = 0.09) -> float:
    zeta = _zeta(z, obukhov_length_m)
    pm = phi_m(zeta)
    return float(u_star**2 * (pm - zeta) / (np.sqrt(c_mu) * pm))


def dissipation(
    z: float, u_star: float, obukhov_length_m: float = 0.0, kappa: float = 0.41
) -> float:
    zeta = _zeta(z, obukhov_length_m)
    return float((u_star**3 / (kappa * z)) * (phi_m(zeta) - zeta))


def eddy_diffusivity(k: float, eps: float, c_mu: float = 0.09, sc_t: float = 0.7) -> float:
    return float(c_mu * k * k / (eps * sc_t))


def turbulence_scales(k: float, eps: float, speed: float, c_mu: float = 0.09) -> tuple[float, float, float]:
    """(u', L_t, tau_L): rms velocity fluctuation, integral length, Lagrangian time."""
    u_prime = float(np.sqrt(2.0 * k / 3.0))
    length = float(c_mu**0.75 * k**1.5 / eps)
    tau = length / (abs(speed) + u_prime)
    return u_prime, length, tau


# --------------------------------------------------------------------------- #
# Assembled flow field
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Flow:
    """Velocity and turbulence on a grid, for one realization."""

    u: np.ndarray  # (ny, nx) m/s
    v: np.ndarray  # (ny, nx) m/s
    k_t: np.ndarray  # (ny, nx) eddy diffusivity for the mean scalar, m2/s
    speed: float  # mean wind speed at flight height, m/s
    direction_rad: float
    u_star: float
    tke: float
    dissipation: float
    u_prime: float
    length_scale: float
    tau_lagrangian: float

    def max_grid_peclet(self, dx: float) -> float:
        return float(np.max(np.hypot(self.u, self.v) * dx / self.k_t))


def _meander(grid: Grid2D, speed: float, direction_rad: float, wavenumber: float,
             amplitude: float, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Cummins Eq. 2, the hand-imposed meandering plume, in a flow-aligned frame.

    Reconstructed from the paper (the published text interleaves the two vector
    components) and verified divergence-free analytically:
        u' = -(U/2) sqrt(3) cos(a y') (x' + 0.1 L) / (1.1 L)
        v' =  (U/2) [ sqrt(3) sin(a y') / (1.1 L a) + 1 ]
    du'/dx' + dv'/dy' = 0 identically.

    This is a decreed geometry, not a turbulence model, and in Cummins it produces
    the largest single effect in the paper (contact probability 22% -> 38% upwind,
    35% -> 57% crosswind). It is off by default and exists only so that comparison
    case can be run. The reconstruction has not been checked against their figure.
    """
    theta = direction_rad
    ex = np.array([np.sin(theta), -np.cos(theta)])  # flow-aligned frame, +y' along flow
    ey = np.array([np.cos(theta), np.sin(theta)])
    x, y = grid.mesh()
    cx, cy = 0.5 * grid.lx, 0.5 * grid.ly
    scale = max(grid.lx, grid.ly)
    xp = (x - cx) * ex[0] + (y - cy) * ex[1] + 0.5 * scale
    yp = (x - cx) * ey[0] + (y - cy) * ey[1] + 0.5 * scale
    a = wavenumber
    up = -(speed / 2.0) * np.sqrt(3.0) * np.cos(a * yp + phase) * (xp + 0.1 * scale) / (1.1 * scale)
    vp = (speed / 2.0) * (np.sqrt(3.0) * np.sin(a * yp + phase) / (1.1 * scale * a) + 1.0)
    up *= amplitude
    return up * ex[0] + vp * ey[0], up * ex[1] + vp * ey[1]


def realize(grid: Grid2D, cfg, rng: np.random.Generator | None = None) -> Flow:
    """Build one flow realization.

    With rng=None the nominal (unperturbed) flow is returned. With an rng, the
    wind direction and friction velocity are jittered, which is what makes replicate
    fields independent -- Cummins reused a single realization across all replicates,
    so their error bars measure only mosquito stochasticity (audit 4.1).
    """
    w = cfg.wind
    z = cfg.domain.flight_height_m
    u_star = w.u_star_m_s
    direction = w.direction_rad
    phase = 0.0
    if rng is not None:
        if w.realization.u_star_rel_sd > 0:
            u_star *= float(np.exp(rng.normal(0.0, w.realization.u_star_rel_sd)))
        if w.realization.direction_sd_rad > 0:
            direction += float(rng.normal(0.0, w.realization.direction_sd_rad))
        if w.meander.enabled and w.realization.meander_phase == "uniform":
            phase = float(rng.uniform(0.0, 2.0 * np.pi))

    if u_star <= 0.0:
        # Windless (indoor) level of the Foppa factorial. Transport is molecular
        # plus a floor of residual mixing; there is no shear to set K_t.
        zero = grid.zeros()
        k_t = np.full((grid.ny, grid.nx), cfg.co2.molecular_diffusivity_m2_s)
        return Flow(
            u=zero, v=zero.copy(), k_t=k_t, speed=0.0, direction_rad=direction,
            u_star=0.0, tke=0.0, dissipation=0.0, u_prime=0.0,
            length_scale=float(min(grid.lx, grid.ly)), tau_lagrangian=np.inf,
        )

    speed = wind_speed(z, u_star, w.z0_m, w.obukhov_length_m, w.kappa_vk)
    k = tke(z, u_star, w.obukhov_length_m, w.c_mu)
    eps = dissipation(z, u_star, w.obukhov_length_m, w.kappa_vk)
    k_t_scalar = eddy_diffusivity(k, eps, w.c_mu, w.sc_t) + cfg.co2.molecular_diffusivity_m2_s
    u_prime, length, tau = turbulence_scales(k, eps, speed, w.c_mu)

    if w.meander.enabled and w.meander.kind != "none":
        u, v = _meander(
            grid, speed, direction, w.meander.wavenumber_rad_m, w.meander.amplitude, phase
        )
    else:
        u = np.full((grid.ny, grid.nx), speed * np.cos(direction))
        v = np.full((grid.ny, grid.nx), speed * np.sin(direction))

    return Flow(
        u=u,
        v=v,
        k_t=np.full((grid.ny, grid.nx), k_t_scalar),
        speed=speed,
        direction_rad=direction,
        u_star=u_star,
        tke=k,
        dissipation=eps,
        u_prime=u_prime,
        length_scale=length,
        tau_lagrangian=tau,
    )
