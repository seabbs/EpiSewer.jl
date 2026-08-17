# Sampling components: integrated outlier detection (`MeasurementOutliers`).

@doc raw"""
    MeasurementOutliers{M, S, T} <: AbstractObservationModel

Integrated outlier detection for a concentration time series (the
`outliers_estimate` component in EpiSewer).

Outliers are modelled as independent **additive spikes** on the expected series:

```math
\tilde{Y}_t = Y_t + \text{scale} \cdot \epsilon_t, \qquad
\epsilon_t \sim \text{GEV}(\mu, \sigma, \xi)
```

The spikes are i.i.d. draws from a generalised extreme value distribution, so
the component is nothing more than an
[`Ascertainment`](@ref ComposableTuringIDModels.Ascertainment) over an
[`IID`](@ref ComposableTuringIDModels.IID) latent process with an additive
`transform`. This matches the R package, which adds
`load_mean * ε_t / flow_median` to the expected concentration
(`inst/stan/EpiSewer_main.stan`, the `additive outlier component` line) and
scores `ε_t` with `gev_lpdf` under the default prior
`GEV(μ = 0, σ = 2e-8, ξ = 4)` (`R/model_sampling.R`, `outliers_estimate`).

The extreme right tail (`ξ = 4`) is what makes the tiny scale meaningful: the
median spike is ~1.7e-8 while the 99% quantile is ~0.49, so a typical day is
untouched and a rare day can absorb up to about half a case-equivalent of load
before the transmission dynamics have to explain it. Stan declares `ε_t`
non-negative; here the GEV's own support (`-σ/ξ = -5e-9` upwards) is used
unaltered, which admits spikes below zero that are ~8 orders of magnitude
smaller than the 99% quantile.

# Fields
- `model`: the wrapped observation model, scoring the spiked expected series.
- `spike`: the prior on the per-day spike `ε_t`, in units of `scale`.
- `scale`: the expected-series-unit equivalent of one unit of spike, i.e. the R
  package's `load_mean / flow_median` when the component sits on the
  concentration scale. Because an `Ascertainment` `transform` cannot see
  sampled parameters, this is a **construction-time constant**: set it from the
  load-per-case prior *median* divided by the median flow rather than from the
  sampled load per case. That approximation fixes the spike's units at the
  prior median instead of tracking the posterior.

# Placement
Put it immediately inside [`FlowNormalize`](@ref EpiSewer.FlowNormalize), so the
spike is added after the flow division and therefore lands on the concentration
scale, exactly where Stan adds it:

    scale = exp(lpc_prior_median) / flow_median
    FlowNormalize(MeasurementOutliers(LogNormalError(); scale = scale))

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = EpiSewer.MeasurementOutliers(NormalError(); scale = 100.0)
model = as_turing_model(m, [100.0, missing, 500.0], fill(100.0, 3))
```
"""
struct MeasurementOutliers{M <: AbstractObservationModel, S, T} <:
    AbstractObservationModel
    "The wrapped observation model."
    model::M
    "Prior on the per-day spike `ε_t`."
    spike::S
    "Expected-series-unit equivalent of one unit of spike."
    scale::T
end

function MeasurementOutliers(
        model::AbstractObservationModel;
        spike = GeneralizedExtremeValue(0.0, 2.0e-8, 4.0),
        scale = 1.0,
    )
    return MeasurementOutliers(model, spike, scale)
end

# The equivalent composition: an additive IID spike on the expected series.
_spiked(m::MeasurementOutliers) = Ascertainment(
    m.model, IID(m.spike);
    transform = (Y_t, eps) -> Y_t .+ m.scale .* eps,
    latent_prefix = "outliers",
)

"""
    as_turing_model(m::MeasurementOutliers, y_t, Y_t)

Score a series against the spiked observation model: draw `ε_t` i.i.d. from
`m.spike`, add `m.scale .* ε_t` to the expected series `Y_t`, and delegate to
the wrapped model. `missing` entries are handled by the wrapped model.

Returns `(; y_t, expected)`, where `expected` is the spiked series.
"""
@model function as_turing_model(m::MeasurementOutliers, y_t, Y_t)
    inner ~ as_turing_submodel(_spiked(m), y_t, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end
