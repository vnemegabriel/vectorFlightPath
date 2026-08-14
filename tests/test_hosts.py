import numpy as np
import pytest

from vfp import config as cfgmod, hosts as hostmod


def test_grid_patch_is_centred_and_correctly_spaced():
    cfg = cfgmod.load("cummins")
    host_set = hostmod.build(cfg)
    assert host_set.n == 9
    assert np.allclose(host_set.positions_m.mean(axis=0), cfg.hosts.patch_center_m)
    span = host_set.positions_m.max(axis=0) - host_set.positions_m.min(axis=0)
    assert np.allclose(span, 2 * cfg.hosts.spacing_m)


def test_two_patches_are_tagged_and_placed_crosswind_of_each_other():
    """Same y, so neither patch sits in the other's wake and the attack-abatement
    comparison is not confounded by shadowing."""
    cfg = cfgmod.load("cummins_foppa")
    host_set = hostmod.build(cfg)
    assert host_set.patch_sizes() == (9, 1)
    first = host_set.positions_m[host_set.patch_id == 0]
    second = host_set.positions_m[host_set.patch_id == 1]
    assert first[:, 1].mean() == pytest.approx(second[:, 1].mean())
    assert abs(first[:, 0].mean() - second[:, 0].mean()) > 3.0


def test_single_host_sits_at_the_patch_centre():
    cfg = cfgmod.load("device_bernier")
    host_set = hostmod.build(cfg)
    assert host_set.n == 1
    assert np.allclose(host_set.positions_m[0], cfg.hosts.patch_center_m)


def test_host_too_close_to_the_wall_is_refused():
    """A capture disc clipped by the domain edge would silently lower the contact
    rate for that host, which is exactly the kind of geometry artefact the
    domain-invariance metric exists to rule out."""
    cfg = cfgmod.load("cummins").replace(**{"hosts.patch_center_m": (0.2, 5.0)})
    with pytest.raises(ValueError, match="contact radius"):
        hostmod.build(cfg)


def test_capacity_is_carried_through():
    cfg = cfgmod.load("cummins").replace(**{"hosts.capacity": 3})
    assert hostmod.build(cfg).capacity == 3
    assert hostmod.build(cfgmod.load("cummins")).capacity == 0
