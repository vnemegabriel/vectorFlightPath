import numpy as np
import pytest

from vfp import config as cfgmod
from vfp.fields import abl
from vfp.grid import Grid2D

KAPPA = 0.41
C_MU = 0.09
SC_T = 0.7


def test_log_law_inverts_cummins_wind():
    """Cummins' large-scale wind U2 = 0.2 m/s at 1 m over z0 = 0.01 m."""
    u_star = abl.u_star_from_speed(0.2, z=1.0, z0=0.01)
    assert u_star == pytest.approx(0.017806, rel=1e-4)
    assert abl.wind_speed(1.0, u_star, 0.01) == pytest.approx(0.2)
    assert cfgmod.load("cummins").wind.u_star_m_s == pytest.approx(u_star, rel=1e-4)


def test_reference_neutral_values():
    u_star = 0.0178
    k = abl.tke(1.0, u_star)
    eps = abl.dissipation(1.0, u_star)
    k_t = abl.eddy_diffusivity(k, eps)
    u_prime, length, tau = abl.turbulence_scales(k, eps, abl.wind_speed(1.0, u_star, 0.01))
    assert k == pytest.approx(1.056e-3, rel=1e-3)
    assert eps == pytest.approx(1.376e-5, rel=1e-3)
    assert k_t == pytest.approx(1.043e-2, rel=1e-3)
    assert u_prime == pytest.approx(0.02653, rel=1e-3)
    assert length == pytest.approx(0.4099, rel=1e-3)
    assert tau == pytest.approx(1.810, rel=1e-3)


def test_eddy_diffusivity_identity_in_neutral_conditions():
    """C_mu k^2 / (eps Sc_t) must reduce to kappa u* z / Sc_t.

    The local-equilibrium closure is only self-consistent if this holds. A sign or
    exponent slip anywhere in tke/dissipation would otherwise pass unnoticed,
    because both functions are individually plausible.
    """
    for z in (0.25, 1.0, 3.0):
        for u_star in (0.005, 0.0178, 0.12):
            k = abl.tke(z, u_star)
            eps = abl.dissipation(z, u_star)
            assert abl.eddy_diffusivity(k, eps) == pytest.approx(KAPPA * u_star * z / SC_T)


def test_neutral_tke_is_height_independent():
    u_star = 0.0178
    assert abl.tke(0.5, u_star) == pytest.approx(abl.tke(5.0, u_star))
    assert abl.tke(1.0, u_star) == pytest.approx(u_star**2 / np.sqrt(C_MU))


def test_stability_functions_are_continuous_at_neutral():
    for f in (abl.phi_m, abl.psi_m):
        assert f(np.array(-1e-9)) == pytest.approx(f(np.array(1e-9)), abs=1e-7)
    assert abl.phi_m(np.array(0.0)) == pytest.approx(1.0)
    assert abl.psi_m(np.array(0.0)) == pytest.approx(0.0, abs=1e-12)


def test_stability_shifts_wind_in_the_expected_direction():
    """Stable air suppresses mixing, so the same u* drives a faster wind at height."""
    u_star = 0.0178
    stable = abl.wind_speed(1.0, u_star, 0.01, obukhov_length_m=10.0)
    neutral = abl.wind_speed(1.0, u_star, 0.01, obukhov_length_m=0.0)
    unstable = abl.wind_speed(1.0, u_star, 0.01, obukhov_length_m=-10.0)
    assert unstable < neutral < stable
    assert abl.tke(1.0, u_star, -10.0) > abl.tke(1.0, u_star, 0.0)


def test_realize_uniform_flow_matches_configured_direction():
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    assert flow.speed == pytest.approx(0.2, rel=2e-3)
    assert np.allclose(flow.u, 0.0, atol=1e-15)  # direction is +y
    assert np.allclose(flow.v, flow.speed)
    assert flow.max_grid_peclet(cfg.numerics.dx_m) == pytest.approx(0.48, rel=0.05)


def test_grid_peclet_stays_below_the_central_scheme_limit_for_every_preset():
    """Central differencing is only non-oscillatory while Pe_dx < 2.

    This is the check that justifies not using upwind, and therefore the check
    that keeps numerical diffusivity at zero instead of 88-1100x molecular.
    """
    for name in ("cummins", "cummins_foppa", "device_bernier"):
        cfg = cfgmod.load(name)
        grid = Grid2D.from_config(cfg)
        pe = abl.realize(grid, cfg).max_grid_peclet(cfg.numerics.dx_m)
        assert pe < cfg.numerics.max_grid_peclet, f"{name}: Pe_dx = {pe:.2f}"


def test_meander_is_divergence_free():
    """Cummins Eq. 2 is a decreed kinematic field; at minimum it must conserve mass."""
    cfg = cfgmod.load("cummins").replace(**{"wind.meander.enabled": True})
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    dudx, _ = grid.gradient(flow.u)
    _, dvdy = grid.gradient(flow.v)
    interior = (slice(2, -2), slice(2, -2))
    divergence = np.abs(dudx + dvdy)[interior].max()
    assert divergence < 1e-6 * flow.speed / grid.dx


def test_meander_actually_meanders():
    cfg = cfgmod.load("cummins").replace(**{"wind.meander.enabled": True})
    grid = Grid2D.from_config(cfg)
    straight = abl.realize(grid, cfg.replace(**{"wind.meander.enabled": False}))
    meandering = abl.realize(grid, cfg)
    assert np.std(meandering.u) > 0.1 * straight.speed
    assert np.allclose(np.std(straight.u), 0.0)


def test_realizations_differ_but_are_reproducible():
    """Independent wind realizations are what defect 4.1 requires."""
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    a = abl.realize(grid, cfg, np.random.default_rng(0))
    b = abl.realize(grid, cfg, np.random.default_rng(1))
    a_again = abl.realize(grid, cfg, np.random.default_rng(0))
    assert a.u_star != b.u_star and a.direction_rad != b.direction_rad
    assert a.u_star == a_again.u_star
    assert np.array_equal(a.v, a_again.v)


def test_windless_realization_falls_back_to_molecular_diffusion():
    """The indoor level of the Foppa factorial: no shear, so no eddy diffusivity."""
    cfg = cfgmod.load("cummins_foppa").replace(**{"wind.u_star_m_s": 0.0})
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg, np.random.default_rng(0))
    assert flow.speed == 0.0
    assert np.allclose(flow.u, 0.0) and np.allclose(flow.v, 0.0)
    assert np.allclose(flow.k_t, cfg.co2.molecular_diffusivity_m2_s)
