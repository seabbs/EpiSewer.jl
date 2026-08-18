# [Model components](@id model-components)

This package replicates the model functionality of the
[EpiSewer](https://github.com/adrian-lison/EpiSewer) R package — a Bayesian
generative model for estimating the effective reproduction number `R_t` and
other transmission indicators from wastewater measurements — using composable
components from the EpiAware ecosystem
([`ComposableTuringIDModels.jl`](https://epiaware.org/ComposableTuringIDModels.jl)
and [`EpiAwareADTools.jl`](https://epiaware.org/EpiAwareADTools.jl)).

The original EpiSewer model is organised into six modules.
The table below lists every model component, its role, and the ecosystem component that provides the same functionality.

## Component mapping

| EpiSewer module | EpiSewer component (R name) | Role in model | Provided by |
|---|---|---|---|
| measurements | `concentrations_observe` | Score the concentration measurements against the expected concentration | An observation error model wrapping the latent expected concentration (`NormalError`, `LogNormalError`) |
| measurements | `concentrations_observe_partitions` | Score the positive dPCR partition counts with a binomial likelihood instead of the concentration | `DigitalPCRError` in `src/measurements.jl`, a `TransformObservationModel` applying the Poisson partition law `p_t = 1 - exp(-exp(Y_t))` over a `BinomialError` |
| measurements | `noise_estimate` | Measurement noise as a constant coefficient of variation | `LogNormalError` in `src/measurements.jl`, a `LogNormal` whose real-space mean is the expected concentration and whose sd is `cv · Y_t`, so `cv` is the inferred coefficient of variation |
| measurements | `noise_estimate_constant_var` | Measurement noise as a constant variance | `NormalError` with a shared `std` prior |
| measurements | `noise_estimate_dPCR` / `noise_estimate_dPCR_params` | Concentration-dependent CV derived from the dPCR partition model, predicting more relative variation at low concentrations | No ecosystem counterpart; this needs a CV that is a function of the expected concentration |
| measurements | `LOD_assume` / `LOD_estimate_dPCR` | Left-censoring of measurements at the detection limit | `LOD` in `src/measurements.jl`, an `AbstractObservationErrorModel` wrapping an inner error distribution in `Distributions.censored`, with values at the limit scored as a `logcdf`. The limit is assumed rather than derived from the dPCR model |
| sampling | `outliers_estimate` | Integrated outlier detection for measurements | `MeasurementOutliers` in `src/sampling.jl`, an `Ascertainment` over `IID(truncated(GeneralizedExtremeValue(0, 2e-8, 4), 0, Inf))` with an additive transform, adding `scale · ε_t` to the expected concentration |
| sampling | `sample_effects_estimate_weekday` / `sample_effects_estimate_matrix` | Log-linear regression of concentration on sample covariates (weekday, age of sample) | `ascertainment_dayofweek` for the weekday case; the general design-matrix regression has no ecosystem counterpart |
| sewage | `flows_observe` / `flows_assume` | Flow-normalise the shed load into a concentration | `FlowNormalize` in `src/sewage.jl`, which divides the expected **load** by the daily flow and then delegates to the wrapped observation model. The flow is data, passed as `y_t = (y = concentrations, flow = flow)` |
| sewage | `residence_dist_assume` | Convolve the shedding signal through a sewer residence-time distribution | `LatentDelay`, via `model()`'s `residence_dist` |
| shedding | `incubation_dist_assume` | Incubation period, so a shedding profile indexed from symptom onset can be applied to infections | `LatentDelay`, via `model()`'s `incubation_dist` |
| shedding | `shedding_dist_assume` / `shedding_dist_estimate` | Shedding load profile, assumed or inferred | `LatentDelay` with the shedding load distribution, or with an `UncertainDelay` when its parameters carry priors |
| shedding | `load_per_case_assume` / `load_per_case_calibrate` | Load shed per case | `Ascertainment` scaling `I_t` by `exp(lpc)` at the observation stage |
| shedding | `load_variation_estimate` | Individual-level shedding-load overdispersion | `LoadVariation` in `src/shedding.jl`, in the default chain outside the load-per-case step. R's `gamma_sum_log_approx`: a normal draw at the gamma's moments, floored by a softplus |
| infections | `generation_dist_assume` | Generation-time distribution between infections | `Renewal(generation_time = ...)`, which discretises a continuous distribution itself and drops the lag-0 bin |
| infections | `R_estimate_rw`, `R_estimate_gp` and six further smoothers | Flexible `R_t` smoothing | `RandomWalk` for the random walk and `HilbertSpaceGP` / `ExactGP` for the GP, both reachable through `model()`'s `rt`, with `CombineLatentModels` summing GP terms and `TransformLatentModel` applying a link. `AR`, `MA` and `DiffLatentModel` are the nearest analogues of the spline, piecewise, exponential-smoothing and smooth-derivative options |
| infections | `seeding_estimate_constant` / `seeding_estimate_growth` / `seeding_estimate_rw` | Infections over the seeding phase, whose length is the generation interval's horizon, before the renewal recursion can be applied | `Renewal`'s `initialisation`, set through `model()`'s `seeding`, seeds the window at `I₀` decaying at the growth rate implied by `R₀`, a fixed exponential. That covers the intercept of R's random walk over the seeding phase; the walk itself needs a custom renewal step whose `recurrent_step` overrides `renewal_init_window` |
| infections | `infection_noise_estimate` | Stochastic infections, negative-binomial around the renewal expectation | `InfectionNoise` in `src/infections.jl`, an `AbstractRenewalModifier` whose `apply_modifier` returns a non-centred draw with the negative-binomial variance at the renewal expectation, which is the incidence the recursion then carries forward |
| forecast | `horizon_assume` | Probabilistic forecast of `R_t`, infections and concentrations | `forecast`, which extends each draw's latent innovations over the horizon and predicts through the observation model |
| forecast | `damping_assume` | Exponential damping of the forecast `R_t` trend, so extrapolated transmission levels off | No ecosystem counterpart |

R's `noise_estimate` offers four distribution families for the concentration likelihood (gamma by default, then log-normal, truncated normal and normal).
`LogNormalError` is the log-normal one.

## Composition

Most of the wastewater model is existing `ComposableTuringIDModels.jl` components.
The infection process (`Renewal`), `R_t` smoothing (`RandomWalk`, `HilbertSpaceGP`), discrete delays (`LatentDelay`) and observation error (`NormalError`, `NegativeBinomialError`) all come from there.

Two of the wastewater-specific pieces are compositions with no new struct at all.
Load-per-case calibration is an `Ascertainment` scaling `I_t` by `exp(lpc)`.
The sewer residence time is a `LatentDelay`, and R's default `residence_dist = c(1)` is a point mass at same-day arrival, so an identity convolution.

This package adds seven `ComposableTuringIDModels.jl`-compatible structs.
Three are compositions of ecosystem pieces:

- `MeasurementOutliers` is an `Ascertainment` over `IID(truncated(GEV(...)))`
  with an additive transform.
- `DigitalPCRError` is a `TransformObservationModel` applying the cloglog-inverse
  link over a `BinomialError`.
- `LOD` wraps an inner error model's distribution in `Distributions.censored`.

Four provide behaviour the ecosystem does not.
`FlowNormalize` reads the daily flow out of the observation-data contract.
`LogNormalError` is a relative-noise error family, parameterised by a
coefficient of variation.
`InfectionNoise` is a renewal modifier, drawing infections with the
negative-binomial variance at the renewal expectation and feeding the draw
forward through the scan.
`LoadVariation` draws the realised shedding load around the expected one, with
the variance of a sum of individual loads.

`InfectionNoise` resolves to an `InfectionNoiseDraws` once its standard normals
are drawn, which is the form the renewal scan steps through.

## The default chain and what sits outside it

`EpiSewer.model()` composes a `Renewal` infection process, with a summed
Gaussian-process `R_t` prior and stochastic infections, observed through the
chain incubation delay → individual-level load variation → per-case load →
shedding delay → flow division → outlier spikes → log-normal noise.
The [getting started](@ref getting-started) page reads both stages off the
assembled model and shows how to replace either.

One part of R's model tree sits outside that assembly.

**The random walk over the seeding phase.** R's `seeding_estimate_rw` runs a
random walk on log infections across the seeding phase, whose length is the
generation interval's horizon.
`model()`'s `seeding` supplies the intercept of that walk, and `Renewal` seeds
the window at `I₀` decaying at the rate implied by `R₀`.
A walk there is a custom renewal step whose `recurrent_step` overrides
`renewal_init_window`.

R's `R_estimate_gp` sums two Matérn-3/2 GPs, a short-term one at 21 ± 3.5 days
with magnitude 0.125 and a long-term one at 84 ± 7 days with magnitude 0.25, then
maps the sum through an `inv_softplus` link.
The default here is the same sum, built by `CombineLatentModels` and mapped
through `softplus_link` with `TransformLatentModel`.
The link is unbounded above; R's `R_max` applies only to its alternative
`scaled_logit` link.
A `HilbertSpaceGP` measures its length scale in standard deviations of the
standardised time index rather than in days, so `gp_length_scale` converts R's
prior in days at a given series length and `model()`'s `n_gp` sets the length the
default assumes.

`residence_dist_assume` needs no wiring either way.
R's default `residence_dist = c(1)` is a point mass at same-day arrival, so `model()`'s `residence_dist` defaults to `nothing` and the wrapper is absent.
Pass `residence_dist` for a non-trivial residence time.

## Discretisation

EpiSewer ships its own discretisation helpers (`get_discrete_gamma` and friends), which assign the mass beyond a right-truncation point to the last bin.
Here, delay inputs are handed to `Renewal` and `LatentDelay` as continuous distributions and those components discretise them.
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
