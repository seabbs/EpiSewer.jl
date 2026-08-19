# Sampling components: integrated outlier detection (`MeasurementOutliers`).

@doc raw"""
    MeasurementOutliers{M, S, T, A} <: AbstractObservationModel

Integrated outlier detection for a concentration time series (the
`outliers_estimate` component in EpiSewer).

Outliers are modelled as independent **additive spikes** on the expected series:

```math
\tilde{Y}_t = Y_t + \text{scale} \cdot \epsilon_t, \qquad
\epsilon_t \sim \text{GEV}(\mu, \sigma, \xi)
```

The spikes are i.i.d. draws from a generalised extreme value distribution
truncated at zero. The component is an `Ascertainment` over an `IID` latent
process with an additive `transform`. This matches the R package,
which adds `load_mean * ε_t / flow_median` to the expected concentration
(`inst/stan/EpiSewer_main.stan`, the `additive outlier component` line) and
scores `ε_t` with `gev_lpdf` under the prior `GEV(μ = 0, σ = 2e-8, ξ = 4)`
(`R/model_sampling.R`, `outliers_estimate`).

The truncation is not cosmetic. Stan declares `vector<lower=0> epsilon` and
scores it with an unnormalised `gev_lpdf`, so the distribution it samples is the
GEV restricted to `[0, ∞)`. The untruncated GEV puts 36.8% of its mass on
negative spikes — an outlier component that *reduces* the expected concentration
— which Stan's parameter space forbids outright.

The extreme right tail (`ξ = 4`) is what makes the tiny scale meaningful: the
median spike is ~2.4e-7 while the 99% quantile is ~3.1, so a typical day is
untouched and a rare day can absorb a few case-equivalents of load before the
transmission dynamics have to explain it.

Note that R's own docstring claims the 99% quantile is "below the load
equivalent of 1 case", which holds for the untruncated GEV (0.49) but not for
the distribution its Stan code actually samples (3.09). R's documentation and
R's model disagree here; this follows the model.

# Fields
- `model`: the wrapped observation model, scoring the spiked expected series.
- `spike`: the prior on the per-day spike `ε_t`, in units of `scale`.
- `scale`: the expected-series-unit equivalent of one unit of spike, i.e. the R
  package's `load_mean / flow_median` when the component sits on the
  concentration scale. Both of those are fixed quantities in Stan — `load_mean`
  is declared in the data block and `flow_median_log` is transformed data — so a
  construction-time constant is faithful to R rather than an approximation of
  it. `model()` fixes the load per case for the same reason, so the two agree.
  Passing a *prior* for the load per case instead makes this an approximation:
  an `Ascertainment` `transform` cannot see sampled parameters, so `scale` is
  set from that prior's median while the scaling itself varies.
- `spiked`: the composed `Ascertainment`, built once here rather than inside
  `as_turing_model`. `Ascertainment`'s constructor validates `transform` with
  `hasmethod`, which lowers to a `Core._hasmethod` foreigncall that Mooncake
  has no rule for, so constructing it in the model body would put that call on
  the differentiated path. Derived from the other three fields; pass them and
  let the constructor build it.

# Placement
Put it immediately inside [`FlowNormalize`](@ref EpiSewer.FlowNormalize), so the
spike is added after the flow division and therefore lands on the concentration
scale, exactly where Stan adds it:

    scale = load_per_case / flow_median
    FlowNormalize(MeasurementOutliers(LogNormalError(); scale = scale))

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = EpiSewer.MeasurementOutliers(NormalError(); scale = 100.0)
model = as_turing_model(m, [100.0, missing, 500.0], fill(100.0, 3))
```
"""
struct MeasurementOutliers{M <: AbstractObservationModel, S, T, A} <:
    AbstractObservationModel
    "The wrapped observation model."
    model::M
    "Prior on the per-day spike `ε_t`."
    spike::S
    "Expected-series-unit equivalent of one unit of spike."
    scale::T
    "The composed `Ascertainment` adding the spike to the expected series."
    spiked::A

    function MeasurementOutliers(
            model::M, spike::S, scale::T
        ) where {M <: AbstractObservationModel, S, T}
        # An additive IID spike on the expected series. Composed here, not in
        # `as_turing_model`, to keep `Ascertainment`'s `hasmethod` check off
        # the differentiated path.
        spiked = Ascertainment(
            model, IID(spike);
            transform = (Y_t, eps) -> Y_t .+ scale .* eps,
            latent_prefix = "outliers",
        )
        return new{M, S, T, typeof(spiked)}(model, spike, scale, spiked)
    end
end

# `spiked` is derived from the other three fields, so the struct has four
# fields and its inner constructor takes three. Accessors rebuilds a struct by
# calling its constructor with every field, which without this method throws a
# `MethodError` and makes the whole chain unreachable to `@set` and `modify`.
# The derived field is recomputed rather than trusted, so it cannot go stale.
MeasurementOutliers(model, spike, scale, _spiked) =
    MeasurementOutliers(model, spike, scale)

function MeasurementOutliers(
        model::AbstractObservationModel;
        # Truncated at zero because Stan declares `vector<lower=0> epsilon` and
        # scores it with an unnormalised `gev_lpdf`; untruncated, 36.8% of the
        # prior mass would sit on negative spikes.
        spike = truncated(GeneralizedExtremeValue(0.0, 2.0e-8, 4.0), 0.0, Inf),
        scale = 1.0,
    )
    return MeasurementOutliers(model, spike, scale)
end

"""
    as_turing_model(m::MeasurementOutliers, y_t, Y_t)

Score a series against the spiked observation model: draw `ε_t` i.i.d. from
`m.spike`, add `m.scale .* ε_t` to the expected series `Y_t`, and delegate to
the wrapped model. `missing` entries are handled by the wrapped model.

Returns `(; y_t, expected)`, where `expected` is the spiked series.
"""
@model function as_turing_model(m::MeasurementOutliers, y_t, Y_t)
    inner ~ as_turing_submodel(m.spiked, y_t, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end
