import numpy as np
import pytest

from vfp import config as cfgmod, hosts as hostmod
from vfp.fields import abl, mean as meanmod, variance as varmod
from vfp.fields.abl import Flow
from vfp.grid import Grid2D


def _quiescent_flow(grid, k_t, tke, eps):
    """Turbulence with no mean motion: production balances dissipation locally."""
    zero = grid.zeros()
    return Flow(
        u=zero, v=zero.copy(), k_t=np.full((grid.ny, grid.nx), k_t), speed=0.0,
        direction_rad=0.0, u_star=0.0, tke=tke, dissipation=eps,
        u_prime=float(np.sqrt(2 * tke / 3)), length_scale=0.41, tau_lagrangian=1.81,
    )


def test_production_dissipation_balance_is_exact():
    """The analytic check on the closure.

    Homogeneous turbulence, uniform mean gradient G, no advection: the transport
    and diffusion terms vanish identically and

        c'2 = K_t G^2 k / (C_d eps)

    holds to machine precision. Everything else in this module is a perturbation
    of that balance, so if it is wrong nothing downstream can be right.
    """
    k_t, tke, eps, c_d, gradient = 1.0445e-2, 1.0568e-3, 1.3769e-5, 2.0, 10.0
    cfg = cfgmod.load("cummins").replace(**{"wind.c_dissipation": c_d, "wind.sc_variance": 0.7})
    grid = Grid2D(10.0, 10.0, 0.1)
    flow = _quiescent_flow(grid, k_t, tke, eps)
    x, _ = grid.mesh()
    mean_field = 1000.0 + gradient * x  # offset keeps the realizability bound slack

    result = varmod.solve(grid, flow, cfg, mean_field)
    expected = k_t * gradient**2 * tke / (c_d * eps)
    assert np.allclose(result.field, expected, rtol=1e-10)
    assert result.clipped_fraction == 0.0


def test_variance_scales_inversely_with_the_dissipation_constant():
    """c'2 ~ 1/C_d, so the intensity moves only as 1/sqrt(C_d): the C_d in [1.5, 2.5]
    sensitivity sweep can shift the intensity by about 15%, no more."""
    grid = Grid2D(10.0, 10.0, 0.1)
    flow = _quiescent_flow(grid, 1.0445e-2, 1.0568e-3, 1.3769e-5)
    x, _ = grid.mesh()
    mean_field = 1000.0 + 10.0 * x
    base = cfgmod.load("cummins")
    low = varmod.solve(grid, flow, base.replace(**{"wind.c_dissipation": 1.5}), mean_field)
    high = varmod.solve(grid, flow, base.replace(**{"wind.c_dissipation": 3.0}), mean_field)
    assert np.allclose(low.field / high.field, 2.0, rtol=1e-10)


def test_fluctuation_intensity_is_invariant_to_source_strength():
    """c'2 is quadratic in the mean, so the intensity -- and therefore the whole
    intermittency structure -- does not depend on emission rate or mixing depth.
    Those two only slide the mean relative to C0."""
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    positions = hostmod.build(cfg).positions_m
    solver = varmod.make_solver(grid, flow, cfg)
    intensities = []
    for emission in (20.0, 300.0):
        field = meanmod.solve(grid, flow, cfg, meanmod.SourceSpec.hosts(positions, emission))
        intensities.append(varmod.solve(grid, flow, cfg, field, solver).intensity_p50)
    assert intensities[0] == pytest.approx(intensities[1], rel=1e-9)


def test_variance_is_non_negative_and_bounded():
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    field = meanmod.solve(
        grid, flow, cfg, meanmod.SourceSpec.hosts(hostmod.build(cfg).positions_m, 300.0)
    )
    result = varmod.solve(grid, flow, cfg, field)
    assert np.all(result.field >= 0.0)
    assert np.all(result.field <= field * (varmod.PURE_TRACER_PPM - field) + 1e-9)
    assert result.clipped_fraction == 0.0, result.report()


def test_variance_peaks_where_the_mean_gradient_is_steepest():
    """Production is 2 K_t |grad C|^2, so the variance maximum must sit near the
    plume edge and the source, not at the concentration maximum."""
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    field = meanmod.solve(
        grid, flow, cfg, meanmod.SourceSpec.hosts(hostmod.build(cfg).positions_m, 300.0)
    )
    variance = varmod.solve(grid, flow, cfg, field).field
    dcdx, dcdy = grid.gradient(field)
    production = dcdx**2 + dcdy**2
    peak_var = np.unravel_index(np.argmax(variance), variance.shape)
    peak_prod = np.unravel_index(np.argmax(production), production.shape)
    assert np.hypot(peak_var[0] - peak_prod[0], peak_var[1] - peak_prod[1]) * grid.dx < 0.5


@pytest.mark.parametrize("preset", ["cummins", "cummins_foppa", "device_bernier"])
def test_every_preset_produces_a_usable_intermittency_structure(preset):
    """The centreline must have genuine blanks and the plume edge must be more
    intermittent than the centreline. Without both, the variance field is doing no
    work and the model reduces to the smoothed field it was built to replace."""
    cfg = cfgmod.load(preset)
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    host_set = hostmod.build(cfg)
    field = meanmod.solve(
        grid, flow, cfg,
        meanmod.SourceSpec.hosts(host_set.positions_m, cfg.co2.emission_per_host_ml_min),
    )
    result = varmod.solve(grid, flow, cfg, field)
    i_c = cfg.co2.pdf.conditional_intensity

    first = host_set.positions_m[host_set.patch_id == 0]
    cx, cy = first[:, 0].mean(), first[:, 1].mean()
    col = int(round(cx / grid.dx - 0.5))
    row = int(round((cy + 1.0) / grid.dx - 0.5))
    centre_intensity = np.sqrt(result.field[row, col]) / field[row, col]
    gamma_centre = min(1.0, (1 + i_c**2) / (1 + centre_intensity**2))
    # Lower bound guards against a signal so rare the agent never sees it; upper
    # bound against no blanks at all. A single host in a 1.4 m/s wind sits near
    # 0.19 at 1 m, a strongly intermittent but perfectly usable trace.
    assert 0.05 < gamma_centre < 0.95, f"centreline gamma = {gamma_centre:.3f}"
    assert result.intensity_p50 > centre_intensity, "plume edges must exceed the centreline"
    assert result.clipped_fraction == 0.0, result.report()


def test_windless_variance_stays_finite():
    cfg = cfgmod.load("cummins_foppa").replace(
        **{"wind.u_star_m_s": 0.0, "boundary.scalar_lateral": "dirichlet"}
    )
    grid = Grid2D(10.0, 10.0, 0.05)
    flow = abl.realize(grid, cfg)
    assert varmod.dissipation_rate(flow, cfg.wind.c_dissipation) == 0.0
    field = meanmod.solve(
        grid, flow, cfg, meanmod.SourceSpec.hosts(hostmod.build(cfg).positions_m, 20.0)
    )
    result = varmod.solve(grid, flow, cfg, field)
    assert np.all(np.isfinite(result.field)) and np.all(result.field >= 0.0)
