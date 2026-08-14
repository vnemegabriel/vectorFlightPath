import json

import pytest

from vfp import config as cfgmod
from vfp.config import Config, ConfigError


def _without_meta(cfg: Config) -> dict:
    return {k: v for k, v in cfg.to_dict().items() if k != "meta"}


def test_base_preset_matches_dataclass_defaults():
    """base.toml is the published parameter table; the defaults are the safety net.

    Keeping them equal is what stops the documented values and the executed values
    from drifting apart -- which is precisely the failure mode that makes Cummins
    et al. unreimplementable.
    """
    assert _without_meta(cfgmod.load("base")) == _without_meta(Config())


def test_every_preset_loads_and_hashes():
    names = cfgmod.available_presets()
    assert {"base", "cummins", "cummins_foppa", "device_bernier"} <= set(names)
    for name in names:
        cfg = cfgmod.load(name)
        assert len(cfg.hash()) == 16


def test_hash_ignores_meta_but_tracks_physics():
    base = cfgmod.load("base")
    assert base.replace(**{"meta.name": "renamed", "meta.notes": "x"}).hash() == base.hash()
    assert base.replace(**{"numerics.dx_m": 0.05}).hash() != base.hash()
    assert base.replace(**{"co2.pdf.conditional_intensity": 2.0}).hash() != base.hash()


def test_hash_is_stable_across_processes():
    """Artefacts carry this hash, so it must not depend on dict ordering or PYTHONHASHSEED."""
    assert cfgmod.load("cummins").hash() == cfgmod.load("cummins").hash()
    assert len(set(cfgmod.load(n).hash() for n in cfgmod.available_presets())) == len(
        cfgmod.available_presets()
    )


def test_extends_deep_merges_nested_tables():
    """Overriding wind.u_star_m_s must not wipe wind.meander or wind.realization."""
    cfg = cfgmod.load("device_bernier")
    assert cfg.wind.u_star_m_s == pytest.approx(0.1246)
    assert cfg.wind.meander.kind == "cummins_eq2"
    assert cfg.wind.realization.u_star_rel_sd == pytest.approx(0.15)


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "typo.toml"
    path.write_text('[numerics]\ndx_meters = 0.05\n')
    with pytest.raises(ConfigError, match="unknown config key"):
        cfgmod.load(path)


def test_unknown_nested_key_reports_full_path(tmp_path):
    path = tmp_path / "typo2.toml"
    path.write_text('[wind.meander]\nwavenumber = 1.0\n')
    with pytest.raises(ConfigError, match=r"wind\.meander\.wavenumber"):
        cfgmod.load(path)


def test_wrong_type_is_rejected(tmp_path):
    path = tmp_path / "badtype.toml"
    path.write_text('[agents]\nn = 1.5\n')
    with pytest.raises(ConfigError, match="expected an integer"):
        cfgmod.load(path)


def test_circular_extends_is_detected(tmp_path):
    (tmp_path / "a.toml").write_text('[meta]\nextends = "b.toml"\n')
    (tmp_path / "b.toml").write_text('[meta]\nextends = "a.toml"\n')
    with pytest.raises(ConfigError, match="circular"):
        cfgmod.load(tmp_path / "a.toml")


@pytest.mark.parametrize(
    "override, message",
    [
        ({"numerics.dx_m": -1.0}, "must be > 0"),
        ({"agents.s_min_m_s": 2.0}, "s_min_m_s <= s_max_m_s"),
        ({"behavior.co2.beta0": 1.5}, r"beta0 must be in \[0, 1\)"),
        ({"behavior.co2.c0_ppm": 9999.0}, "c0_ppm < c_sat_ppm"),
        ({"behavior.wind.t_cwd_s": (0.9, 0.5)}, "0 < lo <= hi"),
        ({"boundary.mode": "teleport"}, "boundary.mode unknown"),
        ({"hosts.capacity": -1}, "capacity must be >= 0"),
        ({"agents.omega_max_rad_s": 0.0}, "omega_max_rad_s must be > 0"),
    ],
)
def test_physical_constraints(override, message):
    with pytest.raises(ConfigError, match=message):
        cfgmod.load("base").replace(**override)


def test_domain_must_be_integer_cells():
    with pytest.raises(ConfigError, match="integer number of cells"):
        cfgmod.load("base").replace(**{"numerics.dx_m": 0.03})


def test_point_contact_tunneling_is_refused():
    """s_max*dt = 0.075 m against a 0.5 m radius is fine; a 0.05 m radius is not.

    Without the swept-segment test, dt convergence would measure a detection
    artefact rather than behaviour.
    """
    cfg = cfgmod.load("base").replace(**{"contact.swept_segment": False})
    assert cfg.contact.radius_m == 0.5
    with pytest.raises(ConfigError, match="tunnels"):
        cfg.replace(**{"contact.radius_m": 0.05})


def test_replace_rejects_unknown_path():
    with pytest.raises(ConfigError, match="unknown config path"):
        cfgmod.load("base").replace(**{"numerics.nonexistent": 1.0})


def test_config_is_json_serialisable():
    json.dumps(cfgmod.load("cummins").to_dict())
