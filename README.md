# vectorFlightPath

A 2D agent-based model of mosquito host-seeking, built to compute **spatial
isoprotection surfaces** for controlled-release spatial repellent devices.

It reimplements the behavioural architecture of Cummins et al. (2012), *A Spatial
Model of Mosquito Host-Seeking Behavior*, PLoS Comput Biol 8(5): e1002500, with a
set of specific corrections listed below.

> **Status: the field layer is built and verified. The agent model is not yet
> written.** See [Milestones](#milestones). Nothing here yet computes an
> isoprotection surface.

---

## Why this exists

The closest methodological precedent, Bernier et al. (2019), solves only the
chemical half of the problem. It is a pure CFD scalar-dispersion simulation
coupled to mosquitoes **sealed inside stationary mesh pouches**, so its "spatial
protection efficacy" reduces to a single concentration threshold — 0.234 ppm at
24 h, 0.097 ppm at 48 h — calibrated against mortality of immobilised insects and
then contoured as an iso-surface. There is no model of behaviour anywhere in it.

The authors name the gap themselves: behavioural endpoints in open space "are
associated with protective surfaces what makes them more complex to correlate
with a spatial distribution", and they did not attempt it.

This project builds that missing half. The isoprotection metric here is a
**reduction in host-contact rate for a freely flying agent**, not mortality of a
caged one.

---

## Installation

Python 3.10+ (`tomli` is used as the `tomllib` backport below 3.11).

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest tests -q
```

```bash
python -m vfp.cli presets
```

Only `presets` and `show-config` exist so far; the rest of the CLI arrives with
the milestones that need it.

---

## What is corrected, and why

Numbering follows the source audit. Each item is a decision the source either got
wrong or left unstated, not a matter of taste.

| # | In Cummins et al. | Here |
|---|---|---|
| 1.1 | Δx and Δt never reported | Declared in config, convergence-tested, Roache GCI published (M7) |
| 1.2 | Source given as J₀ = 1680 ppm/s, a **per-cell** rate | Physical units (m³/s), mass-conserving bilinear deposition |
| 1.5 | Noise correlation length never declared | Log-law + Monin-Obukhov similarity; Lagrangian OU perturbation |
| 2.1 | Contact rate depends on the size of the simulation box | Intensive metric `A_eff = Ṅ/ρ`, with a domain-invariance test (M7) |
| 2.4 | Neumann on every face, including the inflow | Dirichlet C = 0 at inflow, advective outflow, Neumann lateral |
| 2.6 | No turn-rate limit; an agent can reverse instantly | Explicit `ω_max` clamp per step (M5) |
| 2.7 | Heading drawn uniformly in an angular window | von Mises, κ matched to α by equal resultant length R = sin α/α (M5) |
| 2.8 | **Channel blend uses non-unit vectors** | Product-of-von-Mises composition; direction stays unit (M6) |
| 3.2 | Smoothed field, no intermittency | Scalar-variance transport + PDF sampled at read time |
| 3.8 | Host capacity invoked in the discussion, absent from the code | Implemented, default off, run both ways in validation (M8) |
| 4.1 | One wind realization shared by all 15 replicates | Independent realizations; nested variance decomposition (M6) |
| 4.4 | Result conditional on window and boundary treatment | Boundary is an experimental factor; P reported against T (M6) |

### 2.8 is new — it is not in the published audit

Cummins compose the wind and CO₂ channels as

$$\vec{d} = c\,(\cos\theta_w, \sin\theta_w) + (1-c)\,(\cos\theta_c, \sin\theta_c)$$

a sum of **unit** vectors, so $|\vec d| < 1$ whenever the two channels disagree.
Since $\vec d$ then enters the position update directly, **the agent's effective
speed silently drops when its sensory cues conflict** — an undocumented coupling
between disagreement and flight speed.

The fix is to compose in the proper domain. The product of von Mises densities is
itself von Mises with parameter equal to the vector sum of the resultants:

$$\mathbf{R} = \sum_j w_j \kappa_j (\cos\theta_j, \sin\theta_j), \qquad \theta_{\text{target}} = \arg\mathbf{R}, \qquad \kappa_{\text{total}} = |\mathbf{R}|$$

One heading is drawn from $\text{vM}(\theta_{\text{target}}, \kappa_{\text{total}})$ and the direction vector stays
unit. Speed decouples from direction, so the defect cannot recur. Cummins' $c=1/2$
becomes unit weights, and their $c=1$ is no longer a branch: it is the automatic
limit as $\kappa_{\text{CO}_2} \to 0$ inside a signal blank.

### Why the plume width in the source was accidental

Cummins' Δx is bracketable from J₀ even though they never state it. With
$J_0 = Q\cdot 10^6/(\Delta x\,\Delta y\,h)$ and $h = 1$ m, a human ($Q \approx 0.3$ L/min)
implies Δx ≈ 5.5 cm; a roosting chicken — the actual Foppa host — implies
Δx ≈ 1.4 cm. First-order upwind then carries a numerical diffusivity
$U\Delta x/2$ between $1.4\times10^{-3}$ and $1.7\times10^{-2}$ m²/s, i.e. **88× to
1100× molecular**.

The physically correct eddy diffusivity at z = 1 m with u\* = 0.0178 m/s is
$K_t = \kappa u_* z/\mathrm{Sc}_t = 1.04\times10^{-2}$ m²/s — the *same order*. Their plume
width was set by a discretisation artefact that happened to land near the right
value, with no control over it.

Here $K_t$ is explicit, and at Δx = 0.025 m the grid Péclet number is **0.48**, so
central differencing is non-oscillatory and carries *zero* numerical diffusivity.
`assemble()` refuses to run above the Péclet limit rather than degrading quietly.

---

## The intermittency architecture

Two steady fields are precomputed per wind realization: the mean concentration
$\bar C$ and the scalar variance $\overline{c'^2}$. At each step an agent **samples an
instantaneous value** from a local PDF parameterised by that pair.

With probability $1-\gamma$ the agent sits in a *blank* and reads exactly zero; with
probability $\gamma$ it is inside a filament and reads a lognormal, where

$$\gamma = \min\left(1, \frac{1 + i_c^2}{1 + i^2}\right), \qquad i^2 = \overline{c'^2}/\bar C^2$$

By construction $E[c] = \bar C$ and $\mathrm{Var}[c] = \overline{c'^2}$ exactly.

**This is not cosmetic.** Cummins' own literature review states that sustained
upwind flight occurs in response to an *intermittent* CO₂ signal and not to a
uniform one, and that a wide well-mixed plume can fail to induce the flight a
turbulent plume of equal mean concentration does induce. Their model represents
neither: the agent reads a smoothed field, so the concentration statistics it
experiences have the wrong PDF for a responder whose response is non-linear with
a threshold.

### The mean field does not need to cross threshold

A calibration result worth stating plainly, because it looks like a bug and is
not. With Foppa's chickens, or with a single host in a 1.4 m/s wind, the **mean**
plume peaks below the C₀ = 40 ppm sensing threshold:

| Preset | Host | Mean peak | Centreline γ at 1 m |
|---|---|---|---|
| `cummins` | 9 humans | 326 ppm | 0.73 |
| `cummins_foppa` | chickens | ~22 ppm | 0.73 |
| `device_bernier` | 1 human | 25 ppm | 0.19 |

At a fluctuation intensity near 3, γ ≈ 0.125, so the mean concentration *inside a
filament* is roughly 8× the ensemble mean and threshold crossings happen in
bursts. A model that smoothed the field would report these hosts as undetectable.

Because $\overline{c'^2}$ is quadratic in the mean, the fluctuation **intensity is invariant**
to emission rate and mixing depth — those two only slide the mean relative to C₀
and cannot change the intermittency structure. There is a test for that.

---

## Declared parameters

The point of the exercise. Every number is in `src/vfp/presets/base.toml`, which
is tied to the dataclass defaults by test so documented and executed values
cannot drift.

| Parameter | Value | Basis |
|---|---|---|
| `numerics.dx_m` | 0.025 | Provisional; frozen at M7 with its GCI |
| `numerics.dt_s` | 0.05 | Provisional; frozen at M7 with its GCI |
| `domain.mixing_depth_m` | 1.0 | Equal to flight height. **One value for every preset** — never retuned per case |
| `co2.pdf.conditional_intensity` | 0.5 | See below. Swept 0.5–2.0 |
| `agents.omega_max_rad_s` | 10.0 | Genuinely open; no direct measurement for *Culex* in this regime. Swept 5–20 |
| `behavior.co2.beta0` | 0.00505 | = g₀/g_sat, tabulated. The source uses β₀ generically while its value differs per modality and is never given |
| `behavior.wind.beta0` | 0.0 | = v₀/v_sat, tabulated |
| `wind.c_dissipation` | 2.0 | Standard. Swept 1.5–2.5; variance goes as 1/C_d, so intensity moves only as its square root |

**On `conditional_intensity = 0.5`**, which departs from the initial plan's 1.0:
intermittency only appears where the total intensity exceeds $i_c$. The k-ε closure
gives $i \approx 0.85$ on the plume centreline, so $i_c = 1.0$ puts γ at exactly 1
there — no blanks along the plume at all, which is both contrary to wind-tunnel
traces and self-defeating, since the intermittency is the reason for carrying a
variance field at all.

### Reference ABL values

At z = 1 m, u\* = 0.017806 m/s, z₀ = 0.01 m, neutral — the log-law inversion of
Cummins' U₂ = 0.2 m/s:

```
U = 0.200 m/s      k = 1.056e-3 m²/s²     ε = 1.376e-5 m²/s³
K_t = 1.043e-2 m²/s (650× molecular)      u' = 0.0265 m/s
L_t = 0.410 m      τ_L = 1.81 s
```

---

## Design invariants

Two rules that hold the architecture together. Both are enforced by test.

**`fields/` is agnostic to what is being transported.** `mean.py` and
`variance.py` take a `SourceSpec` and a diffusivity and return a scalar field.
Metofluthrin from a CRD is a second `SourceSpec` at the device location with a
different rate — it needs *no new numerics*. `FieldSet` gains two arrays; nothing
in `fields/` changes.

**The repellent cannot be directional.** The three hooks (M10) can suppress a
channel's gain, raise flight speed and turn rate, and accumulate a knockdown
hazard — but none of them can rotate θ. That is encoded in the types, not in the
documentation, for the reason below.

### Why the repellent is non-directional

The initial audit justified choosing tropotaxis over klinotaxis by the need to
sum a repellent *vector*. The literature does not support that vector existing.

None of Ogoma (2012), Norris & Coats (2017), Stevenson (2018), Flores-Mendoza
(2022) or Bernier (2019) contains a single flight trajectory, turn angle, or
heading distribution — all are counts at fixed sampling points. The most
discriminating result runs the other way: in **Stevenson Exp. 3**, with a human
host present, the repellent *increased* indoor host-seeking by 63% (OR 1.87, CI
1.54–2.25, p<0.001) while mortality tripled. The authors attribute it to
excito-repellency — flight activation, not directed avoidance. Gradient descent
cannot produce that sign. Bernier separately measures a very flat spatial
gradient (40–60% mortality from 0.5 to 2.5 m), consistent with a well-mixed field
rather than a sensed one.

Tropotaxis remains the right choice, but for a different reason: klinotaxis in
the source has no stated base case for the recursion $\theta_n = \theta_{n-1}$ or
$\theta_{n-1}+\pi$, and carries one bit rather than a vector. A faithful
reimplementation is therefore impossible, which is a substantive criticism of the
source and not a scoping decision. **Klinotaxis is deliberately not implemented.**

---

## Milestones

| | Deliverable | State |
|---|---|---|
| M0 | Config tree, presets, CLI | **done** |
| M1 | Grid, ABL similarity relations | **done** |
| M2 | Sparse steady operator, mean CO₂ field | **done** |
| M3 | Scalar variance transport | **done** |
| M4 | Concentration PDF sampler, AR(1) copula | pending |
| M5 | Agent state, kinematics, contact, boundaries | pending |
| M6 | CO₂ channel, circular composition, metrics | pending |
| M7 | Domain invariance, mesh/timestep convergence, **freeze Δx and Δt** | pending |
| M8 | Foppa attack-abatement validation | pending |
| M9 | Isoprotection sweep | pending |
| M10 | Repellent hooks | pending |

M7 is not optional. The invariance and convergence results are what distinguish
this from the source.

---

## Verification

71 tests pass. The ones that matter most:

- **Manufactured solution converges at observed order 2.00** (2.006 / 2.001 /
  2.000 over three grids), so the interior scheme is second order as claimed.
- **Mass balance closes to 1e-15** against a boundary efflux computed
  independently of the solve — this is what checks that the boundary treatment is
  consistent with the interior stencil, which the linear solve itself could never
  reveal.
- **Eddy-diffusivity identity** $C_\mu k^2/(\varepsilon \mathrm{Sc}_t) = \kappa u_* z/\mathrm{Sc}_t$
  holds in the neutral limit, pinning the local-equilibrium closure to similarity
  theory. A sign or exponent slip in `tke` or `dissipation` would otherwise pass
  unnoticed, since both are individually plausible.
- **Production/dissipation balance** $\overline{c'^2} = K_t G^2 k/(C_d \varepsilon)$ exact to machine
  precision in homogeneous turbulence.
- **Emitted mass is resolution-independent**, the fix for the per-cell source.
- **Neumann inflow accumulates 17× more** at the upstream face than Dirichlet, so
  defect 2.4 is demonstrated rather than asserted.

---

## Known open items

- **The windless Foppa level needs an indoor ventilation model.** Falling back to
  molecular diffusion in a closed box gives ~91,000 ppm, because 2D steady
  diffusion with no advective removal is logarithmically divergent. The physical
  model is first-order removal by air exchange, which the operator already
  supports through its reaction term. Not currently reachable — it requires
  explicitly overriding `wind.u_star_m_s` — and is scheduled for M8. The windless
  level itself is non-negotiable: Foppa's experiment was indoors, and Cummins
  themselves note their agreement "is surprising since our simulations included
  wind".
- **Cummins Eq. 2 meander is reconstructed, not verified.** The published text
  interleaves the two vector components. The reconstruction is analytically
  divergence-free and tested as such, but has not been checked against their
  figure. It is off by default.
- **CRD emission rates disagree by a factor of 35**, for the repellent phase.
  Bernier and Stevenson give 0.224 mg/s per device (30% metofluthrin);
  Flores-Mendoza gives 6.444 µg/s for the same device family and formulation.
  Stevenson is additionally inconsistent with itself: its text says 0.224 mg/s
  while its Fig. 10 caption says 1×10⁻⁸ kg/s = 0.01 mg/s, a further factor of 22.
  None acknowledges the others. 0.224 mg/s is taken as authoritative, being the
  only one tied to a stated measurement method.
- **Streams are per-run, not per-agent**, so changing `agents.n` changes every
  agent's draws and N is part of a run's identity. A deliberate trade for a
  10–100× stepping speedup, asserted by test so it is documented rather than
  discovered.

---

## References

- Cummins B, Cortez R, Foppa IM, Walbeck J, Hyman JM (2012). A Spatial Model of
  Mosquito Host-Seeking Behavior. *PLoS Comput Biol* 8(5): e1002500.
- Bernier UR, et al. (2019). A combined experimental-computational approach for
  spatial protection efficacy assessment of controlled release devices.
  *PLoS Negl Trop Dis* 13(3): e0007188.
- Stevenson JC, et al. (2018). Controlled release spatial repellent devices
  (CRDs) as novel tools against malaria transmission.
- Foppa IM, et al. Per-capita feeding rate on solitary versus grouped hosts —
  the 4.27 ratio used as the external validation target.
