# Model components

This package replicates the model functionality of the
[EpiSewer](https://github.com/adrian-lison/EpiSewer) R package — a Bayesian
generative model for estimating the effective reproduction number `R_t` and
other transmission indicators from wastewater measurements — using composable
components from the EpiAware ecosystem
([`ComposableTuringIDModels.jl`](https://epiaware.org/ComposableTuringIDModels.jl),
[`CensoredDistributions.jl`](https://epiaware.org/CensoredDistributions.jl),
[`EpiAwareADTools.jl`](https://epiaware.org/EpiAwareADTools.jl)).

The original EpiSewer model is organised into six modules. The table below
lists every model component, its role, and — where one exists — the
`ComposableTuringIDModels.jl` component that provides the same functionality,
so we reuse existing ecosystem pieces and only implement what is genuinely
missing.

## Component mapping

| EpiSewer module | EpiSewer component (R name) | Role in model | ComposableTuringIDModels / ecosystem counterpart | Status |
|---|---|---|---|---|
| measurements | `concentrations_observe` | Observe concentration measurements with a noise model | Observation error model wrapping a latent expected concentration, e.g. `NormalError` / `PoissonError` / `NegativeBinomialError` | covered |
| measurements | `noise_estimate` | Estimate observation noise (coefficient of variation, dPCR noise) | `NormalError(var)` with an estimated variance prior; dPCR specifics need a bespoke error struct | needs new struct (for the dPCR variant `noise_estimate_dPCR`) |
| measurements | `LOD` (limit of detection) | Left-censoring of measurements below the detection limit | No direct counterpart; requires a censored observation model (`CensoredDistributions` left-censoring / a custom `LOD` error struct) | needs new struct |
| sampling | `outliers_estimate` | Integrated outlier detection for measurements | No direct counterpart; a mixture / contamination observation-error modifier would need building | needs new struct |
| sampling | `sample_effects` | Batch effects (weekday, age-of-sample) | `Stratify` (per-batch latent process) can represent grouped effects | covered (via `Stratify`) |
| sewage | `flows_observe` | Flow-normalise concentrations using daily flow | A latent flow / normalisation step composed before the observation model | needs new struct (flow-normalisation wrapper) |
| sewage | `residence_dist_assume` | Convolve infection-shedding signal through a sewer residence-time distribution | `LatentDelay` (discrete convolution with a PMF) | covered (via `LatentDelay`) |
| shedding | `incubation_dist_assume` | Disease-specific discretised incubation period | `LatentDelay` with a PMF from `CensoredDistributions` (`get_discrete_gamma`) | covered |
| shedding | `shedding_dist_assume` | Shedding load distribution (relative to symptom onset or infection) | `LatentDelay` with the shedding-load PMF; discretisation via `CensoredDistributions` | covered |
| shedding | `load_per_case_calibrate` | Calibrate shed load per case against observed cases | `ImportedCases` / a load-scaling step mapping infections to load per case | needs calibration wrapper |
| shedding | `load_variation_estimate` | Individual-level shedding-load overdispersion | Overdispersion via `NegativeBinomialError` (dispersion estimated) | covered (via `NegativeBinomialError`) |
| infections | `generation_dist_assume` | Generation-time distribution between infections | `Renewal(generation_time = pmf, ...)`; PMF from `CensoredDistributions` (`get_discrete_gamma_shifted`) | covered |
| infections | `R_estimate_gp` / `R_estimate_rw` / `R_estimate_spline` | Flexible `R_t` smoothing | `HilbertSpaceGP` / `ExactGP` (GP), `RandomWalk` (RW), `AR` / spline-like trend for changepoint | covered |
| infections | `seeding_estimate_rw` | Initial infection seeding | `ImportedCases` / seeded initialisation inside `Renewal` | covered |
| infections | `infection_noise_estimate` | Stochastic infection noise (overdispersion) | Overdispersion via `NegativeBinomialError` on observed infections | covered |
| forecast | `forecast` | Probabilistic forecast of `R_t`, infections, concentrations | Roll the latent process forward and predict through the observation model (`predict` on the composed Turing model) | covered (via `predict`) |

## Reuse maximisation

We aim to reuse the existing `ComposableTuringIDModels.jl` components as far
as possible and implement only the genuine gaps. Most of the mechanically
distinct parts of EpiSewer — the infection process (`Renewal`), `R_t` smoothing
(`HilbertSpaceGP` / `RandomWalk`), discrete delays (`LatentDelay`), and
observation error (`NegativeBinomialError` / `NormalError`) — already have
composable counterparts. The components marked **needs new struct** above are
the wastewater-specific pieces (flow normalisation, limit-of-detection
censoring, dPCR noise, outlier mixtures) that have no direct counterpart and
will be implemented as small `ComposableTuringIDModels.jl`-compatible
`Struct`s.

## Discretisation via CensoredDistributions

EpiSewer ships its own custom discretisation helpers (e.g.
`get_discrete_gamma_shifted`, `get_discrete_gamma`). In this package we instead
use [`CensoredDistributions.jl`](https://epiaware.org/CensoredDistributions.jl)
for discretising continuous distributions into the PMFs the `Renewal` and
`LatentDelay` components consume (essentially a daily-interval discretisation
of generation, incubation, and shedding-load distributions).
