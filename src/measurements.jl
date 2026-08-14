module Measurements

# Measurement-module components for the EpiSewer port.
#
# These are `ComposableTuringIDModels`-compatible structs: each concrete
# component is a subtype of an `AbstractComposableModel` role and implements
# the corresponding `as_turing_model` method, so it can be composed with the
# rest of the EpiAware ecosystem.

using Turing
using DynamicPPL
using Distributions
using ComposableTuringIDModels
import ComposableTuringIDModels: as_turing_model
using ComposableTuringIDModels: AbstractObservationModel, AbstractObservationErrorModel,
    generate_observation_error_priors, observation_error, _at

export LOD

"""
    LOD{M <: AbstractObservationErrorModel, T}

A limit-of-detection (LOD) censored observation model wrapping an underlying
continuous observation-error model (e.g. `NormalError()`).

Measurements reported at or below the detection limit are **left-censored**:
their exact value is unknown, only that it lies below `lod`. For such values
the likelihood contribution is the cumulative probability of the underlying
error distribution up to the censoring boundary, `logcdf(dist, lod)`, rather
than the density of an (unobserved) exact value. Observations above `lod` are
treated as exact and scored with the ordinary density. `missing` entries are
sampled predictively (the forecast path).

# Fields
- `error_model`: the underlying observation-error model (e.g. `NormalError()`),
  providing the per-time-point error distribution via `observation_error`.
- `lod::T`: the detection limit; observations `<= lod` are censored.

# Example
```julia
using EpiSewer, Distributions
m = LOD(NormalError(); lod = 50.0)
y = [10.0, 120.0, missing]
Y = fill(100.0, 3)
model = as_turing_model(m, y, Y)
```
"""
struct LOD{M <: AbstractObservationErrorModel, T} <: AbstractObservationModel
    error_model::M
    lod::T
end

LOD(error_model::AbstractObservationErrorModel; lod::Real) = LOD(error_model, lod)

# Convenience constructor: a normal-error LOD model with a prior on the
# observation-noise standard deviation (matching EpiSewer's noise-observation
# default).
LOD(; lod::Real = 0.0, std = HalfNormal(0.1)) = LOD(NormalError(; std = std), lod)

@model function as_turing_model(m::LOD, y_t, Y_t)
    # Sample the inner error model's priors (e.g. observation-noise σ) through
    # the standard submodel seam so they compose like any other prior.
    priors ~ to_submodel(
        generate_observation_error_priors(m.error_model, y_t, Y_t), false
    )

    pad_Y_t = Y_t .+ 1.0e-6

    # Extract the observation series. A `MissingObservations` carrier is rebuilt
    # into a plain `Vector{Union{Missing,Float64}}` for scoring simplicity; the
    # concrete carrier is not needed because the branch decisions below are made
    # on observed (non-AD) values. A top-level `missing` (predictive simulation)
    # becomes a length-`Y_t` vector of `missing`.
    y = y_t isa NamedTuple ? y_t.y : y_t
    if y isa ComposableTuringIDModels.MissingObservations
        y = map(
            (v, p) -> p ? v : missing,
            y.value,
            y.present,
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

        # Per-time-point underlying error distribution.
        dist = observation_error(
            m.error_model,
            pad_Y_t[i],
            map(p -> _at(p, i), values(priors))...,
        )
        vn = DynamicPPL.VarName{:y_t}(DynamicPPL.Index((idx,), NamedTuple()))

        if ismissing(obs_i)
            # Predictive / forecast draw.
            val, varinfo = DynamicPPL.tilde_assume!!(
                context, dist, vn, y, varinfo
            )
            scored[idx] = val
        elseif obs_i <= m.lod
            # Left-censored: only known to be below the detection limit.
            # Scoring `true` from `Bernoulli(cdf(dist, lod))` contributes exactly
            # `log(cdf(dist, lod)) = logcdf(dist, lod)`, keeping the censored
            # contribution on the same AD-safe `tilde_observe!!` hot path as an
            # exact observation.
            cens_dist = Bernoulli(cdf(dist, m.lod))
            _, varinfo = DynamicPPL.tilde_observe!!(
                context, cens_dist, true, vn, y, varinfo
            )
            scored[idx] = obs_i
        else
            # Exact observation.
            _, varinfo = DynamicPPL.tilde_observe!!(
                context, dist, obs_i, vn, y, varinfo
            )
            scored[idx] = obs_i
        end
    end

    __varinfo__ = varinfo
    return (; y_t = identity.(scored), expected = Y_t)
end

end # module Measurements
