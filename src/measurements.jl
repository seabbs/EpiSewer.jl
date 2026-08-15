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

A dPCR-specific observation-noise model.

In digital PCR, the sample is partitioned into small reaction chambers
(partitions). The measured positive partition count follows a binomial
distribution: `positive_partitions_t ~ Binomial(total_partitions_t, p_t)`
where `p_t = 1 - exp(-λ_t)` and `λ_t` is the expected copies per partition
derived from the concentration.

The expected series `Y_t` passed into the model should be the **log expected
copies per partition**: `λ_t = exp(Y_t)` — i.e. `Y_t` is on a real line while
the implied positive-copy rate `exp(Y_t)` stays positive. If the upstream
expected series is on the concentration scale (gc/mL), convert it to expected
copies per partition by incorporating the dilution factor and partition volume
before passing `Y_t`.

This component replicates EpiSewer's `noise_estimate_dPCR` likelihood: the
dPCR assay reads out the number of positive partitions, which is binomial
about the expected positives given the copies-per-partition rate.

# Fields
- `total_partitions::T`: the valid partition count per measurement (one per
  time point), threaded into the per-time-point binomial.

# Examples
```julia
dpcr = DigitalPCRError([1000, 1000, 1000])
y = [10, 25, missing]          # observed positive partition counts
Y = log.([0.01, 0.02, 0.03])  # log expected copies per partition
mdl = as_turing_model(dpcr, y, Y)
rand(mdl)
```
"""
struct DigitalPCRError{T <: AbstractVector{<:Integer}} <: AbstractObservationErrorModel
    total_partitions::T
end

@model function generate_observation_error_priors(
        obs_model::DigitalPCRError, y_t, Y_t
    )
    return (; total_partitions = obs_model.total_partitions)
end

function observation_error(m::DigitalPCRError, Y_t, total_partitions_t)
    p_t = 1.0 - exp(-exp(Y_t))
    # Clamp p_t to (0,1) so tiny expected-lambda values do not error.
    return Binomial(total_partitions_t, clamp(p_t, eps(), 1 - eps()))
end

end # module Measurements
