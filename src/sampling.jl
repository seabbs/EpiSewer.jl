# Sampling components: integrated outlier detection (`MeasurementOutliers`).

@doc raw"""
    MeasurementOutliers{M <: AbstractObservationErrorModel, C, S} <: AbstractObservationModel

Integrated outlier detection for a concentration time series (the
`outliers_estimate` component in EpiSewer).

Each observation is modelled as a two-component mixture
`y_t ~ (1 - ε_t)·main(Y_t) + ε_t·outlier(Y_t)`: the *main* component is the
wrapped error model's distribution about the expected value, and the *outlier*
component is a wide, heavy-tailed distribution that downweights contaminated
observations. The per-time-point contamination probability `ε_t` has its own
prior (`Beta(1, 30)` by default). The mixture is scored in closed form per time
point, so `ε_t` is integrated out analytically and no discrete latent is
sampled — keeping gradients smooth.

# Fields
- `error_model`: the wrapped error model (e.g. `NormalError()`), the main
  per-time-point error distribution via `observation_error`.
- `contamination_prob`: the (prior on the) per-time-point outlier probability
  `ε_t`.
- `outlier_scale`: how much more variable the outlier component is than the
  main one — a `HalfNormal` prior or a fixed inflation factor.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = EpiSewer.MeasurementOutliers(NormalError())
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

For each time point `i`: draw the contamination probability `ε_i` and outlier
scale from their priors (through the `_at` seam, so a constant or a process
works); a `missing` entry is sampled predictively from the main error
distribution; an observed entry contributes the closed-form log of the
two-component mixture.

Returns `(; y_t, expected)`.
"""
@model function as_turing_model(m::MeasurementOutliers, y_t, Y_t)
    p ~ as_turing_submodel(m.contamination_prob, length(Y_t); prefix = true)
    s ~ as_turing_submodel(m.outlier_scale, length(Y_t); prefix = true)
    priors ~ to_submodel(
        generate_observation_error_priors(m.error_model, y_t, Y_t), false
    )

    pad_Y_t = Y_t .+ 1.0e-6

    y = y_t isa NamedTuple ? y_t.y : y_t
    if y isa MissingObservations
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

        dist = observation_error(
            m.error_model,
            pad_Y_t[i],
            map(pp -> _at(pp, i), values(priors))...,
        )
        p_i = _at(p, i)
        s_i = _at(s, i)

        vn = DynamicPPL.VarName{:y_t}(DynamicPPL.Index((idx,), NamedTuple()))

        if ismissing(obs_i)
            val, varinfo = DynamicPPL.tilde_assume!!(
                context, dist, vn, y, varinfo
            )
            scored[idx] = val
        else
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

# The wide outlier component: a Normal about the main location with the scale
# inflated by `outlier_scale` (floored at 1 so a tiny main σ stays "wide").
function _outlier_dist(dist::Distribution, scale::Real)
    mu = mean(dist)
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

Distributions.pdf(mix::OutlierMixture, x::Real) = exp(Distributions.logpdf(mix, x))
