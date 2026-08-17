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
| measurements | `noise_estimate` / `noise_estimate_dPCR` | Estimate observation noise (coefficient of variation, dPCR noise) | `LogNormalError` in `src/measurements.jl` — a `LogNormal` whose real-space mean is the expected concentration and whose sd is `σ · Y_t`, so `σ` is the inferred CV; `DigitalPCRError` (binomial partition-count model) for the dPCR variant | covered (implemented) |
| measurements | `LOD` (limit of detection) | Left-censoring of measurements below the detection limit | `LOD` in `src/measurements.jl` — an `AbstractObservationErrorModel` wrapping a `Distributions.censored`-based left-truncated error distribution (data reported at the LOD score as `logcdf`) | covered (implemented) |
| sampling | `outliers_estimate` | Integrated outlier detection for measurements | `MeasurementOutliers` in `src/sampling.jl` — an `Ascertainment` over `IID(GeneralizedExtremeValue(0, 2e-8, 4))` with an additive transform, adding `scale · ε_t` to the expected concentration | covered (implemented) |
| sampling | `sample_effects` | Batch effects (weekday, age-of-sample) | `Stratify` (per-batch latent process) can represent grouped effects | covered (via `Stratify`) |
| sewage | `flows_observe` | Flow-normalise concentrations using daily flow | `FlowNormalize` in `src/sewage.jl` — divides the expected **load** by the daily flow to get expected concentrations, then delegates to the wrapped observation model. The flow is data, passed as `y_t = (y = concentrations, flow = flow)` | covered (implemented) |
| sewage | `residence_dist_assume` | Convolve infection-shedding signal through a sewer residence-time distribution | `LatentDelay` (discrete convolution with a PMF) | covered (via `LatentDelay`) |
| shedding | `incubation_dist_assume` | Disease-specific discretised incubation period | `LatentDelay` with a PMF from `CensoredDistributions` | covered |
| shedding | `shedding_dist_assume` | Shedding load distribution (relative to symptom onset or infection) | `LatentDelay` with the shedding-load PMF; discretisation via `CensoredDistributions` | covered |
| shedding | `load_per_case_calibrate` | Calibrate shed load per case against observed cases | `Ascertainment` (observation-stage `I_t .* exp(lpc)` scaling, in the default chain) | covered (via `Ascertainment`) |
| shedding | `load_variation_estimate` | Individual-level shedding-load overdispersion | An extra latent scale on the expected load, e.g. an `Ascertainment` over `HierarchicalNormal`; not a count family, since the likelihood here is continuous | not in the default chain |
| infections | `generation_dist_assume` | Generation-time distribution between infections | `Renewal(generation_time = pmf, ...)`; PMF from `CensoredDistributions` | covered |
| infections | `R_estimate_gp` / `R_estimate_rw` / `R_estimate_spline` | Flexible `R_t` smoothing | `HilbertSpaceGP` / `ExactGP` (GP), `RandomWalk` (RW), `AR` / spline-like trend for changepoint | covered |
| infections | `seeding_estimate_rw` | Initial infection seeding | `ImportedCases` / seeded initialisation inside `Renewal` | covered |
| infections | `infection_noise_estimate` | Stochastic infection noise (overdispersion) | Noise on the latent infection path, e.g. a `HierarchicalNormal` innovation inside the `Renewal` `rt` process | not in the default chain |
| forecast | `forecast` | Probabilistic forecast of `R_t`, infections, concentrations | Roll the latent process forward and predict through the observation model (`predict` on the composed Turing model) | covered (via `predict`) |

## Reuse maximisation

We aim to reuse the existing `ComposableTuringIDModels.jl` components as far
as possible and implement only the genuine gaps. Most of the mechanically
distinct parts of EpiSewer — the infection process (`Renewal`), `R_t` smoothing
(`HilbertSpaceGP` / `RandomWalk`), discrete delays (`LatentDelay`), and
observation error (`NegativeBinomialError` / `NormalError`) — already have
composable counterparts. The wastewater-specific pieces (flow normalisation,
limit-of-detection censoring, dPCR noise, outlier spikes, load-per-case
calibration) had no direct counterpart in the ecosystem and were implemented as
small `ComposableTuringIDModels.jl`-compatible `Struct`s (`FlowNormalize`,
`LOD`, `LogNormalError`, `DigitalPCRError`, `MeasurementOutliers`) — see the
status column above. Load-per-case calibration reuses `Ascertainment` from the
ecosystem.

## Known gaps

One kind of gap is recorded in the status column, from the 2026-08-17 review
pass (see [LLM-Assisted Development Process](@ref)).

**Not in the default chain.** `EpiSewer.model()` composes a `Renewal` with
random-walk `R_t` and the load → shedding delay → flow division → log-normal
noise chain. The R package's own default model tree additionally uses
`R_estimate_gp` (a Gaussian process rather than a random walk),
`infection_noise_estimate`, `load_variation_estimate`,
`residence_dist_assume` and `outliers_estimate`. Every one of those is a
composition of components that already exist (`HilbertSpaceGP`,
`HierarchicalNormal`, a second `LatentDelay`), so closing the gap is a matter
of assembling them rather than writing anything new.

## Discretisation via CensoredDistributions

EpiSewer ships its own custom discretisation helpers. In this package we
instead use [`CensoredDistributions.jl`](https://epiaware.org/CensoredDistributions.jl)
— `double_interval_censored` (primary within-day averaging plus daily
interval censoring) with the tail renormalised by `Distributions.truncated` —
to build the daily PMFs the `Renewal` and `LatentDelay` components consume
(`EpiSewer.example_distributions()`).
