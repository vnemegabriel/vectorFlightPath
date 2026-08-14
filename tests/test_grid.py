import numpy as np
import pytest

from vfp.grid import Grid2D


@pytest.fixture
def grid():
    return Grid2D(lx=10.0, ly=10.0, dx=0.25)


def test_shape_and_centres(grid):
    assert (grid.nx, grid.ny, grid.n) == (40, 40, 1600)
    assert grid.xc[0] == pytest.approx(0.125)
    assert grid.xc[-1] == pytest.approx(9.875)


def test_ravel_matches_stencil_offsets(grid):
    """k +/- 1 and k +/- nx must be the x and y neighbours, or the assembler is wrong."""
    ix, iy = 7, 11
    k = grid.index(ix, iy)
    assert k == iy * grid.nx + ix
    assert grid.index(ix + 1, iy) == k + 1
    assert grid.index(ix, iy + 1) == k + grid.nx


def test_bilinear_is_exact_on_linear_fields(grid):
    x, y = grid.mesh()
    field = 3.0 + 2.0 * x - 5.0 * y
    rng = np.random.default_rng(0)
    px = rng.uniform(grid.dx, grid.lx - grid.dx, 500)
    py = rng.uniform(grid.dy, grid.ly - grid.dy, 500)
    got = grid.sample(field, px, py)
    assert np.allclose(got, 3.0 + 2.0 * px - 5.0 * py)


def test_bilinear_reproduces_cell_centres(grid):
    x, y = grid.mesh()
    field = np.sin(x) * np.cos(y)
    got = grid.sample(field, x.ravel(), y.ravel())
    assert np.allclose(got, field.ravel())


def test_out_of_domain_clamps_rather_than_extrapolating(grid):
    """Agents routinely leave the CO2 region while the flight domain continues."""
    x, y = grid.mesh()
    field = 2.0 * x
    far = np.array([-50.0, 1e4])
    got = grid.sample(field, far, np.array([5.0, 5.0]))
    assert got[0] == pytest.approx(2.0 * grid.xc[0])
    assert got[1] == pytest.approx(2.0 * grid.xc[-1])


def test_weights_partition_unity(grid):
    rng = np.random.default_rng(1)
    w = grid.interp_weights(rng.uniform(-1, 11, 1000), rng.uniform(-1, 11, 1000))
    assert np.allclose(w.w00 + w.w10 + w.w01 + w.w11, 1.0)
    assert np.all(w.w00 >= 0) and np.all(w.w11 >= 0)
    assert np.all(w.k11 < grid.n)


def test_gradient_exact_on_linear_and_second_order_on_quadratic(grid):
    x, y = grid.mesh()
    dfdx, dfdy = grid.gradient(4.0 * x - 7.0 * y)
    assert np.allclose(dfdx, 4.0)
    assert np.allclose(dfdy, -7.0)

    dfdx, dfdy = grid.gradient(x**2 + 3.0 * y**2)
    assert np.allclose(dfdx, 2.0 * x)
    assert np.allclose(dfdy, 6.0 * y)


def test_deposit_conserves_total_mass_at_any_resolution():
    """The fix for defect 1.2.

    Cummins specify the host source as J0 = 1680 ppm/s, a per-cell rate. Refining
    the grid then silently changes the emitted mass, so a mesh-convergence study
    would be comparing different problems. Depositing a physical amount makes the
    total independent of dx.
    """
    positions = np.array([[3.3, 4.7], [5.0, 5.0], [9.94, 0.06]])
    amounts = np.array([1.0, 2.5, 0.25])
    for dx in (0.5, 0.25, 0.125, 0.0625):
        g = Grid2D(10.0, 10.0, dx)
        assert g.deposit(positions, amounts).sum() == pytest.approx(amounts.sum())


def test_deposit_is_adjoint_of_interpolation():
    """<deposit(p, a), f> == <a, sample(f, p)> to machine precision."""
    g = Grid2D(10.0, 10.0, 0.25)
    rng = np.random.default_rng(2)
    pos = rng.uniform(0.5, 9.5, size=(50, 2))
    amounts = rng.normal(size=50)
    x, y = g.mesh()
    f = np.sin(0.7 * x) * np.cos(0.3 * y)
    assert np.sum(g.deposit(pos, amounts) * f) == pytest.approx(
        np.sum(amounts * g.sample(f, pos[:, 0], pos[:, 1]))
    )


def test_deposit_rejects_sources_outside_the_domain():
    g = Grid2D(10.0, 10.0, 0.25)
    with pytest.raises(ValueError, match="inside the domain"):
        g.deposit(np.array([[10.5, 5.0]]), np.array([1.0]))
