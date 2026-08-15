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
| measurements | `noise_estimate` / `noise_estimate_dPCR` | Estimate observation noise (coefficient of variation, dPCR noise) | `NormalError(var)` for CV noise; `DigitalPCRError` (binomial partition-count model) for the dPCR variant — both implemented in `src/measurements.jl` | covered (implemented) |
| measurements | `LOD` (limit of detection) | Left-censoring of measurements below the detection limit | `LOD` in `src/measurements.jl` — an `AbstractObservationErrorModel` wrapping a `Distributions.censored`-based left-truncated error distribution (data reported at the LOD score as `logcdf`) | covered (implemented) |
| sampling | `outliers_estimate` | Integrated outlier detection for measurements | `MeasurementOutliers` in `src/sampling.jl` — a two-component (main + wide contamination) mixture observation-error model with a `Beta` contamination prior | covered (implemented) |
| sampling | `sample_effects` | Batch effects (weekday, age-of-sample) | `Stratify` (per-batch latent process) can represent grouped effects | covered (via `Stratify`) |
| sewage | `flows_observe` | Flow-normalise concentrations using daily flow | `FlowNormalize` in `src/sewage.jl` — rescales observed and expected concentrations by `flow ./ reference_flow` before delegating to the wrapped error model | covered (implemented) |
| sewage | `residence_dist_assume` | Convolve infection-shedding signal through a sewer residence-time distribution | `LatentDelay` (discrete convolution with a PMF) | covered (via `LatentDelay`) |
| shedding | `incubation_dist_assume` | Disease-specific discretised incubation period | `LatentDelay` with a PMF from `CensoredDistributions` | covered |
| shedding | `shedding_dist_assume` | Shedding load distribution (relative to symptom onset or infection) | `LatentDelay` with the shedding-load PMF; discretisation via `CensoredDistributions` | covered |
| shedding | `load_per_case_calibrate` | Calibrate shed load per case against observed cases | `Ascertainment` (observation-stage `I_t .* exp(lpc)` scaling, in the default chain) | covered (via `Ascertainment`) |
| shedding | `load_variation_estimate` | Individual-level shedding-load overdispersion | Overdispersion via `NegativeBinomialError` (dispersion estimated) | covered (via `NegativeBinomialError`) |
| infections | `generation_dist_assume` | Generation-time distribution between infections | `Renewal(generation_time = pmf, ...)`; PMF from `CensoredDistributions` | covered |
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
composable counterparts. The wastewater-specific pieces (flow normalisation,
limit-of-detection censoring, dPCR noise, outlier mixtures, load-per-case
calibration) had no direct counterpart in the ecosystem and were implemented as
small `ComposableTuringIDModels.jl`-compatible `Struct`s (`FlowNormalize`,
`LOD`, `DigitalPCRError`, `MeasurementOutliers`) — see the status column
above. Load-per-case calibration reuses `Ascertainment` from the ecosystem.

## Discretisation via CensoredDistributions

EpiSewer ships its own custom discretisation helpers. In this package we
instead use [`CensoredDistributions.jl`](https://epiaware.org/CensoredDistributions.jl)
— `double_interval_censored` (primary within-day averaging plus daily
interval censoring) with the tail renormalised by `Distributions.truncated` —
to build the daily PMFs the `Renewal` and `LatentDelay` components consume
(`EpiSewer.example_distributions()`).
