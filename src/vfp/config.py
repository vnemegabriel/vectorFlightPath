"""Configuration tree: TOML in, frozen dataclasses out.

Three properties this module guarantees, each one a defect in the source paper:

- Every numeric the model uses is named here and lands in the run artefact. Cummins
  et al. never report dx or dt (defect 1.1) and never tabulate beta0 (defect 1.5).
- Emission is specified in physical units, never as a per-cell rate. Cummins' J0 of
  1680 ppm/s is cell-volume dependent and therefore undefined without dx (defect 1.2).
- Unknown keys are rejected, so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import json
import types
import typing
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path

try:  # 3.11+
    import tomllib
except ModuleNotFoundError:  # 3.10
    import tomli as tomllib  # type: ignore[no-redef]

PRESET_DIR = Path(__file__).parent / "presets"


class ConfigError(ValueError):
    pass


# --------------------------------------------------------------------------- #
# Leaf tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Meta:
    name: str = "unnamed"
    extends: str = ""
    notes: str = ""


@dataclass(frozen=True)
class Domain:
    lx_m: float = 10.0
    ly_m: float = 10.0
    # The 2D layer thickness. A 2D model cannot avoid needing one: it is what
    # converts a host's volumetric emission into a per-area source. Declared here
    # rather than buried, and swept in the sensitivity study.
    mixing_depth_m: float = 1.0
    flight_height_m: float = 1.0

    def __post_init__(self) -> None:
        for name in ("lx_m", "ly_m", "mixing_depth_m", "flight_height_m"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"domain.{name} must be > 0")


@dataclass(frozen=True)
class Numerics:
    dx_m: float = 0.025
    dt_s: float = 0.05
    advection_scheme: str = "central"
    linear_solver: str = "splu"
    max_grid_peclet: float = 2.0

    def __post_init__(self) -> None:
        if self.dx_m <= 0 or self.dt_s <= 0:
            raise ConfigError("numerics.dx_m and numerics.dt_s must be > 0")
        if self.advection_scheme not in ("central", "upwind"):
            raise ConfigError(
                f"numerics.advection_scheme must be 'central' or 'upwind', "
                f"got {self.advection_scheme!r}"
            )
        if self.linear_solver not in ("splu", "bicgstab"):
            raise ConfigError(
                f"numerics.linear_solver must be 'splu' or 'bicgstab', "
                f"got {self.linear_solver!r}"
            )
        if self.max_grid_peclet <= 0:
            raise ConfigError("numerics.max_grid_peclet must be > 0")


@dataclass(frozen=True)
class Meander:
    enabled: bool = False
    kind: str = "cummins_eq2"
    amplitude: float = 1.0
    wavenumber_rad_m: float = 1.5707963267948966

    def __post_init__(self) -> None:
        if self.kind not in ("cummins_eq2", "none"):
            raise ConfigError(f"wind.meander.kind unknown: {self.kind!r}")


@dataclass(frozen=True)
class WindRealization:
    """What varies between independent field realizations.

    Cummins used one single wind realization for all 15 replicates, so their reported
    standard deviations capture only mosquito stochasticity (defect 4.1). These are
    the knobs that make replicate fields genuinely independent.
    """

    direction_sd_rad: float = 0.20
    u_star_rel_sd: float = 0.15
    meander_phase: str = "uniform"

    def __post_init__(self) -> None:
        if self.direction_sd_rad < 0 or self.u_star_rel_sd < 0:
            raise ConfigError("wind.realization spreads must be >= 0")
        if self.meander_phase not in ("uniform", "fixed"):
            raise ConfigError(f"wind.realization.meander_phase unknown: {self.meander_phase!r}")


@dataclass(frozen=True)
class Wind:
    u_star_m_s: float = 0.017806  # log-law inversion of U = 0.2 m/s at z = 1 m, z0 = 0.01 m
    z0_m: float = 0.01
    obukhov_length_m: float = 0.0  # 0 => neutral; <0 unstable; >0 stable
    direction_rad: float = 1.5707963267948966
    kappa_vk: float = 0.41
    c_mu: float = 0.09
    sc_t: float = 0.7
    sc_variance: float = 0.7
    c_dissipation: float = 2.0
    meander: Meander = field(default_factory=Meander)
    realization: WindRealization = field(default_factory=WindRealization)

    def __post_init__(self) -> None:
        if self.u_star_m_s < 0:
            raise ConfigError("wind.u_star_m_s must be >= 0")
        if self.z0_m <= 0:
            raise ConfigError("wind.z0_m must be > 0")
        for name in ("kappa_vk", "c_mu", "sc_t", "sc_variance", "c_dissipation"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"wind.{name} must be > 0")


@dataclass(frozen=True)
class Co2Pdf:
    model: str = "lognormal_intermittent"
    # Conditional (in-plume) fluctuation intensity s/m: the one free parameter of
    # the PDF model, reported as such and swept.
    #
    # Set to 0.5, not 1.0. Intermittency appears only where the total intensity
    # exceeds this value, since gamma = min(1, (1 + i_c^2)/(1 + i^2)). The k-eps
    # variance closure gives i ~ 0.85 on the plume centreline, so i_c = 1.0 puts
    # gamma at exactly 1 there -- no blanks at all along the plume, which is both
    # contrary to wind-tunnel traces and self-defeating, since the intermittency
    # is the reason for carrying a variance field in the first place.
    conditional_intensity: float = 0.5
    copula_ar1: bool = True
    tau_c_s: float | str = "auto"

    def __post_init__(self) -> None:
        if self.model not in ("lognormal_intermittent", "clipped_gamma"):
            raise ConfigError(f"co2.pdf.model unknown: {self.model!r}")
        if self.conditional_intensity <= 0:
            raise ConfigError("co2.pdf.conditional_intensity must be > 0")
        if isinstance(self.tau_c_s, str):
            if self.tau_c_s != "auto":
                raise ConfigError("co2.pdf.tau_c_s must be a number or 'auto'")
        elif self.tau_c_s <= 0:
            raise ConfigError("co2.pdf.tau_c_s must be > 0")


@dataclass(frozen=True)
class Co2:
    molecular_diffusivity_m2_s: float = 1.6e-5
    # PHYSICAL units. Cummins' J0 = 1680 ppm/s is a per-cell rate and therefore
    # meaningless without dx (defect 1.2). 300 mL/min is a human at rest; a
    # roosting chicken is ~20 mL/min. Concentration scales as emission divided by
    # domain.mixing_depth_m, so the two must be chosen together -- see the plume
    # sensory-band test in tests/test_variance_field.py.
    emission_per_host_ml_min: float = 300.0
    background_ppm: float = 0.0
    pdf: Co2Pdf = field(default_factory=Co2Pdf)

    def __post_init__(self) -> None:
        if self.molecular_diffusivity_m2_s <= 0:
            raise ConfigError("co2.molecular_diffusivity_m2_s must be > 0")
        if self.emission_per_host_ml_min < 0:
            raise ConfigError("co2.emission_per_host_ml_min must be >= 0")


@dataclass(frozen=True)
class Hosts:
    layout: str = "grid_patch"
    n_hosts: int = 9
    spacing_m: float = 0.3
    patch_center_m: tuple[float, float] = (5.0, 5.0)
    # Second patch, used by the Foppa attack-abatement geometry.
    patch2_center_m: tuple[float, float] = (0.0, 0.0)
    n_hosts2: int = 0
    # 0 => unlimited (Cummins). Cummins invoked host saturation as an explanation
    # for attack abatement while their code had no such mechanism (defect 3.8).
    capacity: int = 0

    def __post_init__(self) -> None:
        if self.layout not in ("grid_patch", "two_patches", "single", "explicit"):
            raise ConfigError(f"hosts.layout unknown: {self.layout!r}")
        if self.n_hosts < 1:
            raise ConfigError("hosts.n_hosts must be >= 1")
        if self.spacing_m <= 0:
            raise ConfigError("hosts.spacing_m must be > 0")
        if self.capacity < 0:
            raise ConfigError("hosts.capacity must be >= 0 (0 means unlimited)")
        if self.layout == "two_patches" and self.n_hosts2 < 1:
            raise ConfigError("hosts.layout='two_patches' requires n_hosts2 >= 1")


@dataclass(frozen=True)
class WindPerturbation:
    """Lagrangian velocity perturbation, variance 2k/3, timescale tau_L.

    Replaces Cummins' U_r, a spatially white noise field resampled every 2 s whose
    correlation structure was never declared (defect 1.5).
    """

    enabled: bool = True
    model: str = "ou"

    def __post_init__(self) -> None:
        if self.model not in ("ou", "none"):
            raise ConfigError(f"agents.wind_perturbation.model unknown: {self.model!r}")


@dataclass(frozen=True)
class Agents:
    n: int = 20000
    release: str = "uniform_domain"
    # Preferred over `n`: specifying density rather than count is what makes the
    # domain-size invariance test meaningful (defect 2.1). If > 0, n is derived.
    release_density_m2: float = 0.0
    strategy: str = "crosswind"
    s_min_m_s: float = 0.4
    s_max_m_s: float = 1.5
    # Absent in Cummins, where an agent can reverse heading in a single step.
    omega_max_rad_s: float = 10.0
    groundspeed_cap_m_s: float = 0.0  # 0 => no cap; diagnostic reported regardless
    wind_perturbation: WindPerturbation = field(default_factory=WindPerturbation)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ConfigError("agents.n must be >= 1")
        if self.release not in ("uniform_domain", "downwind_band", "upwind_edge", "disk"):
            raise ConfigError(f"agents.release unknown: {self.release!r}")
        if self.strategy not in ("upwind", "downwind", "crosswind", "random_walk"):
            raise ConfigError(f"agents.strategy unknown: {self.strategy!r}")
        if not 0 < self.s_min_m_s <= self.s_max_m_s:
            raise ConfigError("agents requires 0 < s_min_m_s <= s_max_m_s")
        if self.omega_max_rad_s <= 0:
            raise ConfigError("agents.omega_max_rad_s must be > 0")
        if self.release_density_m2 < 0 or self.groundspeed_cap_m_s < 0:
            raise ConfigError("agents.release_density_m2 and groundspeed_cap_m_s must be >= 0")


@dataclass(frozen=True)
class BehaviorCo2:
    c0_ppm: float = 40.0
    c_sat_ppm: float = 4000.0
    g0_ppm_m: float = 40.0
    g_sat_ppm_m: float = 7900.0
    # Tabulated explicitly. Cummins uses beta0 generically in the ramp while its
    # value differs per modality, recoverable only from ratios (defect 1.5).
    beta0: float = 0.00505
    alpha_min_rad: float = 0.5235987755982988  # pi/6
    alpha_max_rad: float = 3.141592653589793  # pi
    gradient_probe_m: float = 0.025
    # Kept only for exact comparability with Cummins; the architecture does not
    # need it, since the ramp already returns kappa = 0 below threshold.
    hard_threshold: bool = True

    def __post_init__(self) -> None:
        _check_sensing(self, "behavior.co2", "c0_ppm", "c_sat_ppm")
        if self.g0_ppm_m >= self.g_sat_ppm_m:
            raise ConfigError("behavior.co2 requires g0_ppm_m < g_sat_ppm_m")
        if self.gradient_probe_m <= 0:
            raise ConfigError("behavior.co2.gradient_probe_m must be > 0")


@dataclass(frozen=True)
class BehaviorWind:
    v0_m_s: float = 0.0
    v_sat_m_s: float = 0.5
    beta0: float = 0.0
    alpha_min_rad: float = 0.5235987755982988  # pi/6
    alpha_max_rad: float = 1.5707963267948966  # pi/2
    t_cwd_s: tuple[float, float] = (0.5, 0.9)

    def __post_init__(self) -> None:
        _check_sensing(self, "behavior.wind", "v0_m_s", "v_sat_m_s")
        lo, hi = self.t_cwd_s
        if not 0 < lo <= hi:
            raise ConfigError("behavior.wind.t_cwd_s must satisfy 0 < lo <= hi")


@dataclass(frozen=True)
class BehaviorWeights:
    """Channel gains.

    Cummins' c = 1/2 (equal blend in plume-tracking) corresponds to unit weights
    here; c = 1 (wind only, in plume-finding) arises automatically as the limit
    kappa_co2 -> 0 rather than being an explicit branch.
    """

    w_wind: float = 1.0
    w_co2: float = 1.0

    def __post_init__(self) -> None:
        if self.w_wind < 0 or self.w_co2 < 0:
            raise ConfigError("behavior.weights must be >= 0")


@dataclass(frozen=True)
class Behavior:
    co2: BehaviorCo2 = field(default_factory=BehaviorCo2)
    wind: BehaviorWind = field(default_factory=BehaviorWind)
    weights: BehaviorWeights = field(default_factory=BehaviorWeights)


@dataclass(frozen=True)
class Contact:
    radius_m: float = 0.5
    # Point-in-disk testing tunnels once s*dt approaches radius_m, which would
    # contaminate the dt convergence study with a detection artefact.
    swept_segment: bool = True

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ConfigError("contact.radius_m must be > 0")


@dataclass(frozen=True)
class Boundary:
    # Cummins used absorbing walls; their Fig. 4 saturation of directed strategies
    # is a consequence of it, not biology (defect 4.4). Kept available so the
    # artefact can be reproduced and quantified rather than argued about.
    mode: str = "reflect"
    scalar_inflow: str = "dirichlet"  # Cummins used Neumann here (defect 2.4)
    scalar_outflow: str = "advective"
    scalar_lateral: str = "neumann"

    def __post_init__(self) -> None:
        if self.mode not in ("absorb", "reflect", "reinject", "periodic"):
            raise ConfigError(f"boundary.mode unknown: {self.mode!r}")
        if self.scalar_inflow not in ("dirichlet", "neumann"):
            raise ConfigError(f"boundary.scalar_inflow unknown: {self.scalar_inflow!r}")
        if self.scalar_outflow not in ("advective", "neumann", "dirichlet"):
            raise ConfigError(f"boundary.scalar_outflow unknown: {self.scalar_outflow!r}")
        if self.scalar_lateral not in ("neumann", "dirichlet"):
            raise ConfigError(f"boundary.scalar_lateral unknown: {self.scalar_lateral!r}")


@dataclass(frozen=True)
class Run:
    t_final_s: float = 300.0
    n_field_realizations: int = 8
    n_replicates_per_field: int = 1
    master_seed: int = 20260812

    def __post_init__(self) -> None:
        if self.t_final_s <= 0:
            raise ConfigError("run.t_final_s must be > 0")
        if self.n_field_realizations < 1 or self.n_replicates_per_field < 1:
            raise ConfigError("run replication counts must be >= 1")


@dataclass(frozen=True)
class Output:
    dir: str = "artifacts"
    save_trajectories: bool = False
    trajectory_subsample: int = 50

    def __post_init__(self) -> None:
        if self.trajectory_subsample < 1:
            raise ConfigError("output.trajectory_subsample must be >= 1")


@dataclass(frozen=True)
class Repellent:
    enabled: bool = False


@dataclass(frozen=True)
class Config:
    meta: Meta = field(default_factory=Meta)
    domain: Domain = field(default_factory=Domain)
    numerics: Numerics = field(default_factory=Numerics)
    wind: Wind = field(default_factory=Wind)
    co2: Co2 = field(default_factory=Co2)
    hosts: Hosts = field(default_factory=Hosts)
    agents: Agents = field(default_factory=Agents)
    behavior: Behavior = field(default_factory=Behavior)
    contact: Contact = field(default_factory=Contact)
    boundary: Boundary = field(default_factory=Boundary)
    run: Run = field(default_factory=Run)
    output: Output = field(default_factory=Output)
    repellent: Repellent = field(default_factory=Repellent)

    def __post_init__(self) -> None:
        nx = self.domain.lx_m / self.numerics.dx_m
        if abs(nx - round(nx)) > 1e-9 or abs(self.domain.ly_m / self.numerics.dx_m - round(self.domain.ly_m / self.numerics.dx_m)) > 1e-9:
            raise ConfigError(
                f"domain ({self.domain.lx_m} x {self.domain.ly_m} m) must be an integer "
                f"number of cells at dx = {self.numerics.dx_m} m"
            )
        step = self.agents.s_max_m_s * self.numerics.dt_s
        if not self.contact.swept_segment and step > 0.5 * self.contact.radius_m:
            raise ConfigError(
                f"point-contact testing tunnels: s_max*dt = {step:.3f} m against "
                f"contact radius {self.contact.radius_m} m. Enable contact.swept_segment."
            )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        """Short digest of everything except `meta`, stamped into every artefact."""
        payload = {k: v for k, v in self.to_dict().items() if k != "meta"}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return blake2b(canonical.encode(), digest_size=8).hexdigest()

    def replace(self, **overrides: object) -> Config:
        """Dotted-path override, e.g. cfg.replace(**{"numerics.dx_m": 0.05})."""
        data = self.to_dict()
        for dotted, value in overrides.items():
            node = data
            *parents, leaf = dotted.split(".")
            for part in parents:
                if part not in node:
                    raise ConfigError(f"unknown config path: {dotted!r}")
                node = node[part]
            if leaf not in node:
                raise ConfigError(f"unknown config path: {dotted!r}")
            node[leaf] = value
        return _build(Config, data, "")


# --------------------------------------------------------------------------- #
# Construction from plain dicts
# --------------------------------------------------------------------------- #


def _check_sensing(obj: object, path: str, lo_name: str, hi_name: str) -> None:
    lo, hi = getattr(obj, lo_name), getattr(obj, hi_name)
    if lo < 0 or hi <= 0 or lo >= hi:
        raise ConfigError(f"{path} requires 0 <= {lo_name} < {hi_name}")
    a_min, a_max = obj.alpha_min_rad, obj.alpha_max_rad  # type: ignore[attr-defined]
    if not 0 < a_min <= a_max:
        raise ConfigError(f"{path} requires 0 < alpha_min_rad <= alpha_max_rad")
    if not 0 <= obj.beta0 < 1:  # type: ignore[attr-defined]
        raise ConfigError(f"{path}.beta0 must be in [0, 1)")


def _coerce(value: object, hint: object, path: str) -> object:
    origin = typing.get_origin(hint)
    if origin is types.UnionType or origin is typing.Union:
        for arm in typing.get_args(hint):
            if arm is str and isinstance(value, str):
                return value
            if arm is float and isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        raise ConfigError(f"{path}: cannot interpret {value!r} as {hint}")
    if origin is tuple:
        args = typing.get_args(hint)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} values, got {value!r}")
        return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
    if hint is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if hint is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    raise ConfigError(f"{path}: unsupported config type {hint!r}")


def _build(cls: type, data: dict, path: str) -> object:
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(sorted(prefix + u for u in unknown))}. "
            f"Valid keys here: {', '.join(sorted(names))}"
        )
    kwargs: dict[str, object] = {}
    for name, value in data.items():
        hint = hints[name]
        child = f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{child}: expected a table, got {value!r}")
            kwargs[name] = _build(hint, value, child)
        else:
            kwargs[name] = _coerce(value, hint, child)
    return cls(**kwargs)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge so overriding one key in [wind] does not wipe [wind.meander]."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    resolved = path.resolve()
    if resolved in seen:
        chain = " -> ".join(p.name for p in (*seen, resolved))
        raise ConfigError(f"circular `extends` chain: {chain}")
    if not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")
    with resolved.open("rb") as fh:
        data = tomllib.load(fh)
    parent_name = data.get("meta", {}).get("extends", "")
    if not parent_name:
        return data
    parent_path = resolved.parent / parent_name
    parent = _resolve(parent_path, (*seen, resolved))
    merged = _deep_merge(parent, data)
    # `extends` is a property of this file, not of the merged result.
    merged.setdefault("meta", {})["extends"] = parent_name
    return merged


def load(source: str | Path) -> Config:
    """Load a config by preset name or by path, resolving `extends`."""
    path = Path(source)
    if not path.suffix:
        path = PRESET_DIR / f"{source}.toml"
    elif not path.exists() and (PRESET_DIR / path.name).exists():
        path = PRESET_DIR / path.name
    return typing.cast(Config, _build(Config, _resolve(path), ""))


def available_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.toml"))
