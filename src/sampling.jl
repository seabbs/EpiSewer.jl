"""
    EpiSewer.Sampling

Sampling components for the EpiSewer port: outlier detection and
sample-batch effects in wastewater measurements (`MeasurementOutliers`).
"""
module Sampling

# Sampling-module components for the EpiSewer port.
#
# These are `ComposableTuringIDModels`-compatible structs: each concrete
# component is a subtype of an `AbstractComposableModel` role and implements
# the corresponding `as_turing_model` method, so it can be composed with the
# rest of the EpiAware ecosystem.

using Turing: Turing
using DynamicPPL: DynamicPPL, @model, to_submodel
using Distributions: Distributions, Beta, ContinuousUnivariateDistribution,
    Distribution, Normal, mean
using ComposableTuringIDModels: ComposableTuringIDModels, HalfNormal
import ComposableTuringIDModels: as_turing_model, as_turing_submodel
using ComposableTuringIDModels: AbstractObservationModel, AbstractObservationErrorModel,
    generate_observation_error_priors, observation_error, _at

export MeasurementOutliers

@doc raw"""
    MeasurementOutliers{M <: AbstractObservationErrorModel, C, S} <: AbstractObservationModel

Integrated outlier detection for a concentration time series (the
`outliers_estimate` component in EpiSewer).

Independent spikes in the measurements can bias estimates of transmission
dynamics. `MeasurementOutliers` wraps an underlying observation-error model and
adds a per-time-point contamination component, so that each observation is
modelled as arising from a two-component mixture:

```math
y_t \sim (1 - \epsilon_t)\,\mathrm{main}(Y_t) \;+\; \epsilon_t\,\mathrm{outlier}(Y_t)
```

where the *main* component is the wrapped error model's distribution about the
expected value ``Y_t`` and the *outlier* component is a wide (heavy-tailed)
distribution that downweights contaminated observations. The per-time-point
contamination probability ``\epsilon_t`` has its own prior (a `Beta` by
default, so outliers are rare).

The mixture is scored in closed form (AD-smooth) per time point via
``\log\big((1-\epsilon_t)\,p_{\mathrm{main}}(y_t) + \epsilon_t\,p_{\mathrm{outlier}}(y_t)\big)``,
so no discrete latent ``\epsilon_t`` is sampled — the outlier probability is
integrated out analytically, keeping gradients smooth.

# Fields
- `error_model`: the underlying observation-error model (e.g. `NormalError()`),
  providing the main per-time-point error distribution via `observation_error`.
- `contamination_prob`: the (prior on the) per-time-point outlier probability
  ``\epsilon_t``. A `Beta` by default (e.g. `Beta(1, 30)` so outliers are
  rare); a constant or a per-time-point process is also supported through the
  `_at` seam.
- `outlier_scale`: how much more variable the outlier component is than the
  main one — a `HalfNormal` prior or a fixed inflation factor.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = MeasurementOutliers(NormalError())
model = as_turing_model(m, [100.0, missing, 500.0], fill(100.0, 3))
```
"""
struct MeasurementOutliers{
        M <: AbstractObservationErrorModel, C, S,
    } <: AbstractObservationModel
    error_model::M
    contamination_prob::C
    outlier_scale::S
end

function MeasurementOutliers(
        error_model::AbstractObservationErrorModel;
        contamination_prob = Beta(1, 30),
        outlier_scale = HalfNormal(2.0),
    )
    return MeasurementOutliers(error_model, contamination_prob, outlier_scale)
end

@doc raw"""
    as_turing_model(m::MeasurementOutliers, y_t, Y_t)

Score a series against an outlier-mixture likelihood.

For each time point `i`:
- the contamination probability ``\epsilon_i`` and outlier scale are drawn from
  their priors (through the `_at` seam, so a constant or process works);
- a `missing` entry is sampled predictively from the main error distribution
  (the forecast path);
- an observed entry contributes the closed-form log of the two-component
  mixture:
  ``\log\big((1-\epsilon_i)\,p_{\mathrm{main}}(y_i) + \epsilon_i\,p_{\mathrm{outlier}}(y_i)\big)``.

Returns the `(; y_t, expected)` tuple: `y_t` the scored series (observed entries
preserved, blanks filled with fresh draws) and `expected` the pre-error `Y_t`.
"""
@model function as_turing_model(m::MeasurementOutliers, y_t, Y_t)
    # Draw the per-time-point outlier probability and scale through the
    # standard submodel seam, composing like any other prior.
    # Draw the per-time-point outlier probability and scale through the
    # prior seam (`as_turing_submodel`), which composes a bare `Distribution`
    # as a single native scalar draw and a process as a length-`n` submodel.
    p ~ as_turing_submodel(m.contamination_prob, length(Y_t); prefix = true)
    s ~ as_turing_submodel(m.outlier_scale, length(Y_t); prefix = true)

    # Draw the wrapped error model's priors (e.g. observation-noise σ).
    priors ~ to_submodel(
        generate_observation_error_priors(m.error_model, y_t, Y_t), false
    )

    pad_Y_t = Y_t .+ 1.0e-6

    # Extract the observation series, allowing `MissingObservations`, a
    # top-level `missing` and a `NamedTuple` carrying `y`.
    y = y_t isa NamedTuple ? y_t.y : y_t
    if y isa ComposableTuringIDModels.MissingObservations
        y = map(
            (v, ispresent) -> ispresent ? v : missing, y.value, y.present
        )
    elseif ismissing(y)
        y = Vector{Missing}(missing, length(Y_t))
    end
    diff_t = length(y) - length(Y_t)
    @assert diff_t >= 0 "The observation vector must be at least as long as the expected observation vector"

    context = __model__.context
    varinfo = __varinfo__

    scored = Vector{Any}(undef, length(y))
    for i in eachindex(Y_t)
        idx = i + diff_t
        @inbounds obs_i = y[idx]

        # Per-time-point main error distribution.
        dist = observation_error(
            m.error_model,
            pad_Y_t[i],
            map(pp -> _at(pp, i), values(priors))...,
        )

        # Per-time-point outlier probability and scale.
        p_i = _at(p, i)
        s_i = _at(s, i)

        vn = DynamicPPL.VarName{:y_t}(DynamicPPL.Index((idx,), NamedTuple()))

        if ismissing(obs_i)
            # Predictive / forecast draw from the main distribution.
            val, varinfo = DynamicPPL.tilde_assume!!(
                context, dist, vn, y, varinfo
            )
            scored[idx] = val
        else
            # Closed-form mixture log-likelihood (AD-smooth).
            mixture = OutlierMixture(dist, _outlier_dist(dist, s_i), p_i)
            _, varinfo = DynamicPPL.tilde_observe!!(
                context, mixture, obs_i, vn, y, varinfo
            )
            scored[idx] = obs_i
        end
    end

    __varinfo__ = varinfo
    return (; y_t = identity.(scored), expected = Y_t)
end

# The wide outlier component: a Normal centred on the expected value with a
# standard deviation inflated by the outlier scale. `mean(dist)` is the main
# distribution's location (for a Normal the expected concentration), and the
# scale is floored at 1 so a very small main σ still yields a "wide" outlier.
function _outlier_dist(dist::Distribution, scale::Real)
    mu = mean(dist)
    # A scale of `0` (or a fixed constant of `1`) keeps a sensible, finite
    # outlier width even when the main σ is tiny; clamp a non-positive scale to
    # a small positive value so the mixture stays well-defined.
    sigma = max(float(abs(scale)), 1.0)
    return Normal(mu, sigma)
end

# A continuous distribution whose log-pdf is the closed-form two-component
# mixture, so the standard `tilde_observe!!` path scores it AD-smoothly.
struct OutlierMixture{D <: Distribution, S <: Distribution, P <: Real} <:
    ContinuousUnivariateDistribution
    main::D
    outlier::S
    p::P
end

Base.eltype(::Type{<:OutlierMixture}) = Float64
Distributions.partype(::OutlierMixture) = Float64

function Distributions.logpdf(mix::OutlierMixture, x::Real)
    logp_main = log1p(-mix.p) + Distributions.logpdf(mix.main, x)
    logp_out = log(mix.p) + Distributions.logpdf(mix.outlier, x)
    return Distributions.logsumexp((logp_main, logp_out))
end

function Distributions.pdf(mix::OutlierMixture, x::Real)
    return exp(Distributions.logpdf(mix, x))
end

end # module Sampling
