import numpy as np
import pytest

from vfp import config as cfgmod
from vfp.fields import abl, mean as meanmod
from vfp.fields.operator import BCSpec, LinearFieldSolver, OperatorError, boundary_efflux
from vfp.grid import Grid2D

ALL_DIRICHLET = BCSpec("dirichlet", "dirichlet", "dirichlet")
OPEN = BCSpec("dirichlet", "advective", "dirichlet")


def _uniform(grid, u, v, k):
    shape = (grid.ny, grid.nx)
    return np.full(shape, u), np.full(shape, v), np.full(shape, k)


def _manufactured(grid, u, v, k, length):
    """phi = sin^2(pi x/L) sin^2(pi y/L): vanishes on the boundary, so the discrete
    Dirichlet ghost value of zero is exact and the measured order is the interior
    scheme's, uncontaminated by boundary treatment."""
    a = b = np.pi / length
    x, y = grid.mesh()
    f, g = np.sin(a * x) ** 2, np.sin(b * y) ** 2
    fp, gp = a * np.sin(2 * a * x), b * np.sin(2 * b * y)
    fpp, gpp = 2 * a**2 * np.cos(2 * a * x), 2 * b**2 * np.cos(2 * b * y)
    return f * g, u * fp * g + v * f * gp - k * (fpp * g + f * gpp)


def test_manufactured_solution_converges_at_second_order():
    u, v, k, length = 0.3, 0.2, 0.05, 2.0
    errors = []
    for dx in (0.1, 0.05, 0.025):
        grid = Grid2D(length, length, dx)
        solver = LinearFieldSolver(grid, *_uniform(grid, u, v, k), bc=ALL_DIRICHLET)
        exact, source = _manufactured(grid, u, v, k, length)
        errors.append(float(np.sqrt(np.mean((solver.solve(source) - exact) ** 2))))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    assert min(orders) > 1.95, f"observed orders {orders}"


def test_point_source_matches_the_analytic_plume():
    speed, k_t, strength = 0.2, 1.043e-2, 1.0
    worst = {}
    for dx in (0.05, 0.025):
        grid = Grid2D(10.0, 10.0, dx)
        u, v, k = _uniform(grid, speed, 0.0, k_t)
        solver = LinearFieldSolver(grid, u, v, k, bc=OPEN)
        source = grid.deposit(np.array([[5.0, 5.0]]), np.array([strength / grid.cell_area]))
        numeric = solver.solve(source)
        exact = meanmod.analytic_point_source(grid, 5.0, 5.0, strength, speed, k_t)
        x, y = grid.mesh()
        band = (x > 6.0) & (x < 8.0) & (np.abs(y - 5.0) < 1.5) & (exact > 1e-6 * exact.max())
        rel = np.abs(numeric[band] - exact[band]) / exact[band]
        worst[dx] = float(rel.max())
        assert np.median(rel) < 0.005
    # Second order: halving dx should cut the worst-case error by about four.
    assert worst[0.05] / worst[0.025] > 3.0


def test_plume_width_follows_the_gaussian_scaling():
    """sigma = sqrt(2 K x / U) is the slender-plume limit, which drops streamwise
    diffusion; the exact solution is marginally wider, hence the 5% band."""
    speed, k_t = 0.2, 1.043e-2
    grid = Grid2D(10.0, 10.0, 0.025)
    u, v, k = _uniform(grid, speed, 0.0, k_t)
    source = grid.deposit(np.array([[5.0, 5.0]]), np.array([1.0 / grid.cell_area]))
    field = LinearFieldSolver(grid, u, v, k, bc=OPEN).solve(source)
    for downstream in (2.0, 3.0):
        col = int(round((5.0 + downstream) / grid.dx - 0.5))
        profile = field[:, col]
        mass = profile.sum()
        centre = (profile * grid.yc).sum() / mass
        sigma = np.sqrt((profile * (grid.yc - centre) ** 2).sum() / mass)
        assert sigma == pytest.approx(np.sqrt(2 * k_t * downstream / speed), rel=0.05)
        assert centre == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize("dx", [0.1, 0.05, 0.025])
def test_global_mass_balance_closes_exactly(dx):
    """Efflux is computed independently of the solve, so this checks that the
    boundary treatment is consistent with the interior stencil -- something the
    linear solve itself could never reveal."""
    speed, k_t, strength = 0.2, 1.043e-2, 3.7
    grid = Grid2D(10.0, 10.0, dx)
    u, v, k = _uniform(grid, speed, 0.0, k_t)
    source = grid.deposit(np.array([[4.0, 5.0]]), np.array([strength / grid.cell_area]))
    field = LinearFieldSolver(grid, u, v, k, bc=OPEN).solve(source)
    assert boundary_efflux(grid, u, v, k, field, OPEN) == pytest.approx(strength, rel=1e-10)


def test_emitted_mass_is_independent_of_resolution():
    """The fix for defect 1.2.

    Cummins give the host source as J0 = 1680 ppm/s, a per-cell rate, and never
    give dx. Under that formulation refining the grid changes the emitted mass, so
    successive grids in a convergence study solve different problems.
    """
    cfg = cfgmod.load("cummins")
    source = meanmod.SourceSpec.hosts(np.array([[5.0, 5.0], [5.3, 5.0]]), 20.0)
    expected = meanmod.emitted_ppm_m2_per_s(source, cfg.domain.mixing_depth_m)
    for dx in (0.1, 0.05, 0.025):
        grid = Grid2D(10.0, 10.0, dx)
        field = meanmod.source_field(grid, source, cfg.domain.mixing_depth_m)
        assert float(field.sum() * grid.cell_area) == pytest.approx(expected, rel=1e-12)


def test_dirichlet_inflow_holds_the_upstream_face_at_zero():
    speed, k_t = 0.2, 1.043e-2
    grid = Grid2D(10.0, 10.0, 0.05)
    u, v, k = _uniform(grid, speed, 0.0, k_t)
    source = grid.deposit(np.array([[1.0, 5.0]]), np.array([1.0 / grid.cell_area]))
    field = LinearFieldSolver(grid, u, v, k, bc=OPEN).solve(source)
    assert field[:, 0].max() < 1e-3 * field.max()


def test_neumann_inflow_accumulates_where_dirichlet_does_not():
    """Defect 2.4, made concrete.

    Cummins impose zero normal gradient on every face including the inflow. With
    advective influx and no diffusive escape, scalar piles up against the upstream
    wall. At a slow wind, where upstream diffusion reaches the boundary, the two
    treatments differ by a wide margin.
    """
    speed, k_t = 0.02, 1.043e-2
    grid = Grid2D(10.0, 10.0, 0.05)
    u, v, k = _uniform(grid, speed, 0.0, k_t)
    source = grid.deposit(np.array([[1.0, 5.0]]), np.array([1.0 / grid.cell_area]))
    correct = LinearFieldSolver(grid, u, v, k, bc=OPEN).solve(source)
    cummins = LinearFieldSolver(grid, u, v, k, bc=BCSpec("neumann", "advective", "neumann")).solve(source)
    face_ratio = cummins[:, 0].max() / correct[:, 0].max()
    assert face_ratio > 10.0, f"inflow-face concentration ratio only {face_ratio:.1f}"
    assert cummins.sum() > 1.2 * correct.sum()


def test_grid_peclet_guard_refuses_rather_than_degrading():
    grid = Grid2D(10.0, 10.0, 0.5)
    u, v, k = _uniform(grid, 2.0, 0.0, 1.043e-2)
    with pytest.raises(OperatorError, match="grid Peclet"):
        LinearFieldSolver(grid, u, v, k, bc=OPEN, max_grid_peclet=2.0)


def test_upwind_is_available_but_names_its_price():
    """Kept only so the Cummins scheme can be reproduced and its numerical
    diffusivity quantified, never as a silent fallback."""
    grid = Grid2D(10.0, 10.0, 0.5)
    u, v, k = _uniform(grid, 2.0, 0.0, 1.043e-2)
    solver = LinearFieldSolver(grid, u, v, k, bc=OPEN, scheme="upwind")
    source = grid.deposit(np.array([[2.0, 5.0]]), np.array([1.0 / grid.cell_area]))
    assert np.all(solver.solve(source) >= -1e-12)  # upwind is monotone


def test_closed_box_without_a_sink_is_refused():
    """The windless Foppa level. A source in a sealed box has no steady state; the
    error says so instead of returning a silently meaningless field."""
    grid = Grid2D(10.0, 10.0, 0.1)
    u, v, k = _uniform(grid, 0.0, 0.0, 1.6e-5)
    with pytest.raises(OperatorError, match="scalar_lateral"):
        LinearFieldSolver(grid, u, v, k, bc=BCSpec("dirichlet", "advective", "neumann"))


def test_windless_diffusive_field_is_radially_symmetric():
    cfg = cfgmod.load("cummins_foppa").replace(
        **{"wind.u_star_m_s": 0.0, "boundary.scalar_lateral": "dirichlet"}
    )
    grid = Grid2D(10.0, 10.0, 0.05)
    flow = abl.realize(grid, cfg)
    source = meanmod.SourceSpec.hosts(np.array([[5.0, 5.0]]), 20.0)
    field = meanmod.solve(grid, flow, cfg, source)
    x, y = grid.mesh()
    r = np.hypot(x - 5.0, y - 5.0)
    ring = (r > 1.0) & (r < 1.05)
    assert field[ring].std() / field[ring].mean() < 0.01
    assert field.max() > 0


def test_solver_reuses_one_factorisation_across_sources():
    """The economics of the isoprotection sweep: move the host, keep the flow."""
    cfg = cfgmod.load("cummins")
    grid = Grid2D.from_config(cfg)
    flow = abl.realize(grid, cfg)
    solver = meanmod.make_solver(grid, flow, cfg)
    fields = [
        meanmod.solve(grid, flow, cfg, meanmod.SourceSpec.hosts(np.array([[px, 5.0]]), 20.0), solver)
        for px in (3.0, 5.0, 7.0)
    ]
    peaks = [float(np.unravel_index(np.argmax(f), f.shape)[1]) * grid.dx for f in fields]
    assert peaks[0] < peaks[1] < peaks[2]
    for f in fields:
        assert np.all(f >= -1e-9)
