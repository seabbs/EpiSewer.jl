module Sewage

# Sewage-module components for the EpiSewer port.
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

export FlowNormalize

"""
    FlowNormalize{E <: AbstractObservationErrorModel}

Wastewater flow normalization (the `flows_observe` component in EpiSewer).

Raw wastewater concentrations depend on the daily flow through the sampling
site: the same pathogen load produces a higher concentration when the flow is
low. `FlowNormalize` removes this day-to-day flow effect by normalizing both the
observed and the expected concentrations before they are scored against an
underlying observation-error model.

Concentration `c` (gc/mL) and flow `f` (mL/day) combine to a load
`load = c * f` (gc/day). Normalizing to a reference flow `f_ref` gives the
hypothetical concentration were the flow at its reference value:

```math
c_{\\text{norm}} = c \\cdot \\frac{f}{f_{ref}}
```

(the same transformation `plot_concentration(..., normalized = TRUE)` applies
in EpiSewer, defaulting `f_{ref}` to the median flow). With a fixed reference
flow this is a deterministic, linear scaling of both series, so it composes as a
*modifier* wrapping the underlying error model: we flow-normalize the observed
and expected series and delegate scoring to the wrapped error model exactly as
if we had observed the normalized series directly.

# Fields
- `error_model`: the underlying observation-error model (e.g. `NormalError()`),
  operating on the flow-normalized concentrations.
- `flow`: the daily flow series (mL/day), one value per time point, used to
  normalize the concentrations.
- `reference_flow`: the flow value to normalize to (default the median of
  `flow`). Concentrations are scaled by `flow ./ reference_flow`.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
mn = FlowNormalize(NormalError(); flow = [1.0e14, 2.0e14])
model = as_turing_model(mn, [100.0, missing], [100.0, 100.0])
```
"""
struct FlowNormalize{
        E <: AbstractObservationErrorModel, F <: AbstractVector,
        R <: Real,
    } <: AbstractObservationModel
    error_model::E
    flow::F
    reference_flow::R
end

function FlowNormalize(
        error_model::AbstractObservationErrorModel,
        flow::AbstractVector{<:Real};
        reference_flow::Real = median(flow),
    )
    return FlowNormalize(error_model, flow, reference_flow)
end

# Convenience constructor: a normal-error flow-normalization with a prior on
# the observation-noise standard deviation, matching the EpiSewer default.
function FlowNormalize(
        error_model::AbstractObservationErrorModel;
        flow::AbstractVector{<:Real}, reference_flow::Union{Real, Nothing} = nothing
    )
    ref = reference_flow === nothing ? median(flow) : reference_flow
    return FlowNormalize(error_model, flow, ref)
end

"""
    as_turing_model(m::FlowNormalize, y_t, Y_t)

Normalize the observed `y_t` and expected `Y_t` concentrations by
`flow ./ reference_flow` and delegate scoring to the wrapped
`error_model`'s generic observation-error loop.

The flow scale `flow ./ reference_flow` is a fixed, deterministic vector (not a
latent/AD-active parameter), so this is a static reweighting of the two series
before they reach the error model. Missing entries are preserved through the
normalization and handled (sampled predictively) by the wrapped model.

Returns whatever `as_turing_model(m.error_model, y_norm, Y_norm)` returns — the
`(; y_t, expected)` tuple, with `y_t` the flow-normalized scored series and
`expected` the flow-normalized expected series.
"""
@model function as_turing_model(m::FlowNormalize, y_t, Y_t)
    scale = m.flow ./ m.reference_flow

    # Normalize the observed series, preserving `missing` entries. A
    # `MissingObservations` carrier is normalized per-entry on its plain
    # values; a top-level `missing` (predictive simulation) stays `missing`.
    y = y_t isa NamedTuple ? y_t.y : y_t
    if y isa ComposableTuringIDModels.MissingObservations
        y_norm = map(
            (v, p, s) -> p ? (v * s) : missing,
            y.value, y.present, scale,
        )
    elseif ismissing(y)
        y_norm = Vector{Missing}(missing, length(Y_t))
    else
        @assert length(y) == length(scale) ||
            length(y) >= length(Y_t) "flow must be aligned with observations"
        y_norm = y .* scale
    end

    # Normalize the expected series.
    Y_norm = Y_t .* scale

    # Delegate scoring to the wrapped error model's generic loop.
    return as_turing_model(m.error_model, y_norm, Y_norm)
end

end # module Sewage
