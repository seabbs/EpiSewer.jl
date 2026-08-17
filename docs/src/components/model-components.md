# [Model components](@id model-components)

This package replicates the model functionality of the
[EpiSewer](https://github.com/adrian-lison/EpiSewer) R package — a Bayesian
generative model for estimating the effective reproduction number `R_t` and
other transmission indicators from wastewater measurements — using composable
components from the EpiAware ecosystem
([`ComposableTuringIDModels.jl`](https://epiaware.org/ComposableTuringIDModels.jl)
and [`EpiAwareADTools.jl`](https://epiaware.org/EpiAwareADTools.jl)).

The original EpiSewer model is organised into six modules.
The table below lists every model component, its role, and — where one exists — the ecosystem component that provides the same functionality, so we reuse existing pieces and only implement what is genuinely missing.

## Component mapping

| EpiSewer module | EpiSewer component (R name) | Role in model | ComposableTuringIDModels / ecosystem counterpart | Status |
|---|---|---|---|---|
| measurements | `concentrations_observe` | Score the concentration measurements against the expected concentration | An observation error model wrapping the latent expected concentration (`NormalError`, `LogNormalError`) | covered |
| measurements | `concentrations_observe_partitions` | Score the positive dPCR partition counts with a binomial likelihood instead of the concentration | `DigitalPCRError` in `src/measurements.jl` — a `TransformObservationModel` applying the Poisson partition law `p_t = 1 - exp(-exp(Y_t))` over a `BinomialError` | implemented here |
| measurements | `noise_estimate` | Measurement noise as a constant coefficient of variation | `LogNormalError` in `src/measurements.jl` — a `LogNormal` whose real-space mean is the expected concentration and whose sd is `σ · Y_t`, so `σ` is the inferred CV | implemented here |
| measurements | `noise_estimate_constant_var` | Measurement noise as a constant variance | `NormalError` with a shared `std` prior | covered |
| measurements | `noise_estimate_dPCR` / `noise_estimate_dPCR_params` | Concentration-dependent CV derived from the dPCR partition model, predicting more relative variation at low concentrations | None; needs a CV that is a function of the expected concentration | not modelled |
| measurements | `LOD_assume` / `LOD_estimate_dPCR` | Left-censoring of measurements at the detection limit | `LOD` in `src/measurements.jl` — an `AbstractObservationErrorModel` wrapping an inner error distribution in `Distributions.censored`, with values at the limit scored as a `logcdf`. The limit is assumed rather than derived from the dPCR model | implemented here |
| sampling | `outliers_estimate` | Integrated outlier detection for measurements | `MeasurementOutliers` in `src/sampling.jl` — an `Ascertainment` over `IID(truncated(GeneralizedExtremeValue(0, 2e-8, 4), 0, Inf))` with an additive transform, adding `scale · ε_t` to the expected concentration | implemented here, not in the default chain |
| sampling | `sample_effects_estimate_weekday` / `sample_effects_estimate_matrix` | Log-linear regression of concentration on sample covariates (weekday, age of sample) | `ascertainment_dayofweek` covers the weekday case; the general design-matrix regression has no ecosystem counterpart | weekday case covered |
| sewage | `flows_observe` / `flows_assume` | Flow-normalise the shed load into a concentration | `FlowNormalize` in `src/sewage.jl` — divides the expected **load** by the daily flow, then delegates to the wrapped observation model. The flow is data, passed as `y_t = (y = concentrations, flow = flow)` | implemented here |
| sewage | `residence_dist_assume` | Convolve the shedding signal through a sewer residence-time distribution | `LatentDelay`, via `model()`'s `residence_dist` | covered |
| shedding | `incubation_dist_assume` | Incubation period, so a shedding profile indexed from symptom onset can be applied to infections | `LatentDelay`, the outermost wrapper of the default observation chain | covered |
| shedding | `shedding_dist_assume` / `shedding_dist_estimate` | Shedding load profile, assumed or inferred | `LatentDelay` with the shedding load distribution, or with an `UncertainDelay` when its parameters carry priors | covered |
| shedding | `load_per_case_assume` / `load_per_case_calibrate` | Load shed per case | `Ascertainment` scaling `I_t` by `exp(lpc)` at the observation stage | covered |
| shedding | `load_variation_estimate` | Individual-level shedding-load overdispersion | None; needs a Gamma sum over the day's cases | not modelled |
| infections | `generation_dist_assume` | Generation-time distribution between infections | `Renewal(generation_time = ...)`, which discretises a continuous distribution itself and drops the lag-0 bin | covered |
| infections | `R_estimate_rw`, `R_estimate_gp` and six further smoothers | Flexible `R_t` smoothing | `RandomWalk` (what this package uses) and `HilbertSpaceGP` / `ExactGP` for the GP. `AR`, `MA` and `DiffLatentModel` are the nearest analogues of the spline, piecewise, exponential-smoothing and smooth-derivative options, none of which has a direct counterpart | random walk covered |
| infections | `seeding_estimate_constant` / `seeding_estimate_growth` / `seeding_estimate_rw` | Infections over the seeding phase, whose length is the generation interval's horizon, before the renewal recursion can be applied | `Renewal`'s `initialisation` seeds the window at `I₀` decaying at the growth rate implied by `R₀`, a fixed exponential rather than any of R's three | fixed exponential only |
| infections | `infection_noise_estimate` | Stochastic infections, negative-binomial around the renewal expectation | None; needs a renewal modifier | not modelled |
| forecast | `horizon_assume` | Probabilistic forecast of `R_t`, infections and concentrations | `forecast`, which extends each draw's latent innovations over the horizon and predicts through the observation model | covered |
| forecast | `damping_assume` | Exponential damping of the forecast `R_t` trend, so extrapolated transmission levels off | None | not modelled |

R's `noise_estimate` offers four distribution families for the concentration likelihood (gamma by default, then log-normal, truncated normal and normal).
`LogNormalError` is the log-normal one.

## Reuse maximisation

We aim to reuse the existing `ComposableTuringIDModels.jl` components as far
as possible and implement only the genuine gaps. Most of the mechanically
distinct parts of EpiSewer — the infection process (`Renewal`), `R_t` smoothing
(`HilbertSpaceGP` / `RandomWalk`), discrete delays (`LatentDelay`), and
observation error (`NegativeBinomialError` / `NormalError`) — already have
composable counterparts.

Of the wastewater-specific pieces, two need no new component at all:
load-per-case calibration is an `Ascertainment`, and the sewer residence time is
a `LatentDelay` (and by default not even that, since EpiSewer's default
`residence_dist = c(1)` is a point mass at same-day arrival, i.e. an identity
convolution).

The remaining five are `ComposableTuringIDModels.jl`-compatible `Struct`s.
Three are compositions of ecosystem pieces:

- `MeasurementOutliers` is an `Ascertainment` over `IID(truncated(GEV(...)))`
  with an additive transform.
- `DigitalPCRError` is a `TransformObservationModel` applying the cloglog-inverse
  link over a `BinomialError`.
- `LOD` wraps an inner error model's distribution in `Distributions.censored`.

Two implement behaviour the ecosystem does not provide.
`FlowNormalize` reads the daily flow out of the observation-data contract.
`LogNormalError` is a relative-noise error family, parameterised by a
coefficient of variation.

## What the default chain leaves out

`EpiSewer.model()` composes a `Renewal` with random-walk `R_t` and the incubation delay → per-case load → shedding delay → flow division → log-normal noise chain.
Five components of R's own default model tree are absent from it.
Each has a known route through existing ecosystem pieces.

**`R_estimate_gp`.** R sums two Matérn-3/2 Hilbert-space GPs, one long and one short length scale, and an intercept, then maps the sum through a softplus link with a soft upper bound on `R_t`.
`HilbertSpaceGP` provides the GPs, `CombineLatentModels` the sum, and `TransformLatentModel` the link.
This package uses a `RandomWalk` instead.

**`seeding_estimate_rw`.** R runs a random walk on log infections across the seeding phase, whose length is the generation interval's horizon.
`Renewal` instead seeds that window at `I₀` decaying at the rate implied by `R₀`.
Replacing it means a custom renewal step whose `recurrent_step` overrides `renewal_init_window`.

**`infection_noise_estimate`.** R samples infections with negative-binomial overdispersion around the renewal expectation, so the sampled value, not the expectation, is what the next day convolves.
That is a renewal modifier.
`apply_modifier` returns the incidence the recursion carries forward, and `ImportedCases` is the template for a modifier that draws its own priors before the scan.

**`load_variation_estimate`.** R replaces the expected case count with a Gamma sum over that day's cases, giving a population-level CV of `ν / sqrt(cases)` for an individual-level CV of `ν`.
Because that scaling needs the case count, the component has to sit outside the load-per-case step, which is what turns cases into load.

**`outliers_estimate`.** `MeasurementOutliers` implements this, but the default chain does not include it.
Wiring it in means placing it immediately inside `FlowNormalize`, so the spike is added after the flow division and lands on the concentration scale, with `scale` set from the load-per-case prior median over the median flow.

`residence_dist_assume` is not in this list.
EpiSewer's default `residence_dist = c(1)` is a point mass at same-day arrival, so the default chain correctly omits the wrapper.
Pass `residence_dist` to `EpiSewer.model()` for a non-trivial residence time.

## Discretisation

EpiSewer ships its own discretisation helpers (`get_discrete_gamma` and friends), which assign the mass beyond a right-truncation point to the last bin.
This package ships none.
Delay inputs are handed to `Renewal` and `LatentDelay` as continuous distributions and those components discretise them.
They apply primary uniform-window censoring, right-truncation at the horizon `D`, then daily interval censoring, and evaluate the result on the bin left edges before renormalising.
The discretisation itself lives in `ComposableTuringIDModels.jl`, which calls
[`CensoredDistributions.jl`](https://epiaware.org/CensoredDistributions.jl)'s
`double_interval_censored`.
This package does not depend on `CensoredDistributions.jl` directly.

One discretisation path means one place for the conventions to live.
A generation interval has no mass at lag 0, which `Renewal` enforces by dropping that bin and renormalising.
The default generation time is a location-shifted `Gamma` with no mass below lag 1, so that step removes nothing and the renormalisation is exact.

The same keyword arguments also accept an already discretised PMF vector, or an
`UncertainDelay` whose distribution parameters carry priors, in which case the
delay is inferred rather than fixed.
A PMF vector is used verbatim, so a caller passing one owns the no-same-day-transmission convention themselves.
