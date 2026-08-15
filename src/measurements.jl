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
import ComposableTuringIDModels: generate_observation_error_priors, observation_error

export LOD, DigitalPCRError

"""
    LOD{E <: AbstractObservationErrorModel, T}

A limit-of-detection (LOD) censored observation model wrapping an underlying
continuous observation-error model (e.g. `NormalError()`).

Measurements at or below the detection limit are **left-censored**: the exact
value is unknown, only that it lies at or below `lod`. LOD is expressed as a
left-truncated observation-error model using `Distributions.censored`:

- an observed value **above** `lod` is scored with the ordinary density of the
  underlying error distribution (exact observation);
- an observed value **equal to** `lod` is scored with the CDF of the underlying
  error distribution up to the boundary (`logcdf(dist, lod)`), which is the
  correct censored likelihood — the data convention is that censored
  measurements are **reported at the LOD value** (as in EpiSewer's example data);
- a value **strictly below** `lod` scores `-Inf` (impossible under the model).

`missing` entries are handled by the generic `AbstractObservationErrorModel`
loop (sampled predictively, the forecast path).

Implementing `LOD` as an `AbstractObservationErrorModel` (rather than a
hand-rolled `AbstractObservationModel`) means the censoring lives entirely in
the per-time-point error distribution returned by [`observation_error`](@ref),
and the standard scoring loop applies unchanged — no per-branch tilde logic to
maintain.

# Fields
- `error_model`: the underlying continuous observation-error model (e.g.
  `NormalError()`), providing the per-time-point error distribution that is
  left-censored.
- `lod::T`: the detection limit.

# Example
```julia
using EpiSewer, Distributions
m = LOD(NormalError(); lod = 50.0)
y = [50.0, 120.0, missing]   # censored (at LOD), exact, missing
Y = fill(100.0, 3)
model = as_turing_model(m, y, Y)
```
"""
struct LOD{E <: AbstractObservationErrorModel, T} <: AbstractObservationErrorModel
    error_model::E
    lod::T
end

LOD(error_model::AbstractObservationErrorModel; lod::Real) = LOD(error_model, lod)

# Convenience constructor: a normal-error LOD model with a prior on the
# observation-noise standard deviation (matching EpiSewer's noise-observation
# default).
LOD(; lod::Real = 0.0, std = HalfNormal(0.1)) = LOD(NormalError(; std = std), lod)

# The inner error model's priors (e.g. observation-noise σ) are reused as-is:
# the censored distribution is derived from the inner per-time-point error
# distribution, which is a function of those same priors.
generate_observation_error_priors(m::LOD, y_t, Y_t) =
    generate_observation_error_priors(m.error_model, y_t, Y_t)

# Left-censor the underlying per-time-point error distribution at `lod`.
# `censored(dist, lod, Inf)`:
#   - x == lod  -> logcdf(dist, lod)   (censored: reported at the LOD)
#   - x  > lod  -> logpdf(dist, x)     (exact)
#   - x  < lod  -> -Inf                (below the LOD is impossible in the data)
function observation_error(m::LOD, Y_t, priors...)
    return censored(observation_error(m.error_model, Y_t, priors...), m.lod, Inf)
end

"""
    DigitalPCRError{T <: AbstractVector{<:Integer}}

A dPCR-specific observation-noise model (`noise_estimate_dPCR` in EpiSewer).

In digital PCR, the sample is partitioned into small reaction chambers
(partitions). The measured positive partition count follows a binomial
distribution:

```math
\\mathrm{positive\\,partitions}_t \\sim \\mathrm{Binomial}(\\mathrm{total\\,partitions}_t, p_t),
\\qquad p_t = 1 - \\exp(-\\exp(Y_t)),
```

where `Y_t` is the **log expected copies per partition** supplied by the model.
This is the standard dPCR likelihood: the expected copies per partition
``\\lambda_t = \\exp(Y_t)`` maps to the per-partition positivity probability
``p_t = 1 - \\exp(-\\lambda_t)`` via the Poisson partition law.

The component is expressed entirely through ecosystem pieces:
[`TransformObservationModel`](@ref ComposableTuringIDModels.TransformObservationModel)
applies the inverse-complementary-log-log link `x -> 1 .- exp.(-exp.(x))` to the
expected series, and [`BinomialError`](@ref ComposableTuringIDModels.BinomialError)
scores the observed positive-partition counts against the per-time-point trial
totals. The data contract follows `BinomialError`: `y_t` is a `NamedTuple`
`(y = positive_partitions, N = total_partitions)`.

# Fields
- `total_partitions::T`: the valid partition count per measurement (one per
  time point), threaded into the per-time-point binomial.

# Examples
```julia
dpcr = DigitalPCRError([1000, 1000, 1000])
y = (y = [10, 25, missing], N = dpcr.total_partitions)  # observed positives + totals
Y = log.([0.01, 0.02, 0.03])  # log expected copies per partition
mdl = as_turing_model(dpcr, y, Y)
rand(mdl)
```
"""
struct DigitalPCRError{T <: AbstractVector{<:Integer}} <: AbstractObservationModel
    total_partitions::T
end

# The dPCR noise model as a transform-observation + binomial-error composition:
#   Y_t (log copies/partition) --cloglog-inverse--> p_t --Binomial(N_t)--> positives.
# `as_turing_model(m::DigitalPCRError, y_t, Y_t)` builds this composition and
# samples the inner model with the data's trial counts.
_transformed_dpcr(m::DigitalPCRError) = TransformObservationModel(
    BinomialError(),
    x -> 1.0 .- exp.(-exp.(x)),
)

@model function as_turing_model(m::DigitalPCRError, y_t, Y_t)
    # Merge the data NamedTuple: the user supplies (y = positives, N = total_partitions)
    # inside `y_t`. If N is already present, use it; otherwise add the struct's
    # total_partitions as the trial count.
    y = y_t isa NamedTuple ? merge(y_t, (N = m.total_partitions,)) : (y = y_t, N = m.total_partitions)
    # The inner model (TransformObservationModel + BinomialError) scores the
    # transformed expected observations via the binomial likelihood.
    inner ~ as_turing_submodel(_transformed_dpcr(m), y, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end

end # module Measurements
