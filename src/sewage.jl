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
import ComposableTuringIDModels: as_turing_model, as_turing_submodel
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
- `error_model`: the underlying observation model (e.g. `NormalError()`),
  operating on the flow-normalized concentrations. Any
  `AbstractObservationModel` composes here, so the full observation chain
  (e.g. `LatentDelay(Ascertainment(NormalError(), prior), pmf)`) can be
  wrapped by the flow normalization.
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
        E <: AbstractObservationModel, F <: AbstractVector,
        R <: Real,
    } <: AbstractObservationModel
    error_model::E
    flow::F
    reference_flow::R
end

function FlowNormalize(
        error_model::AbstractObservationModel,
        flow::AbstractVector{<:Real};
        reference_flow::Real = median(flow),
    )
    return FlowNormalize(error_model, flow, reference_flow)
end

# Convenience constructor: a normal-error flow-normalization with a prior on
# the observation-noise standard deviation, matching the EpiSewer default.
function FlowNormalize(
        error_model::AbstractObservationModel;
        flow::AbstractVector{<:Real}, reference_flow::Union{Real, Nothing} = nothing
    )
    ref = reference_flow === nothing ? median(flow) : reference_flow
    return FlowNormalize(error_model, flow, ref)
end

"""
    as_turing_model(m::FlowNormalize, y_t, Y_t)

Convert an expected **load** series (`Y_t`, gc/day) into an expected
**concentration** series by dividing by the daily flow, and delegate scoring
to the wrapped `error_model`'s generic observation-error loop.

EpiSewer models the pathogen load at the sampling site (gc/day) and derives
the observed concentration as `concentration_t = load_t / flow_t`. This
component applies that division to the expected series (which the model chain
provides as a load), so the scored expected values are on the observed
concentration scale (gc/mL). The observed series `y_t` passes through
unchanged. The flow is fixed, deterministic data (not a latent/AD-active
parameter); missing entries are preserved and handled (sampled predictively)
by the wrapped model.

Returns whatever `as_turing_model(m.error_model, y_norm, Y_norm)` returns — the
`(; y_t, expected)` tuple, with `y_t` the observed series and `expected` the
flow-converted expected concentration series.
"""
@model function as_turing_model(m::FlowNormalize, y_t, Y_t)
    # The expected series is a load (gc/day); divide by the daily flow to get
    # an expected concentration (gc/mL) on the observed scale. When the
    # expected series is shorter than the flow (e.g. a LatentDelay above has
    # truncated it), align the trailing flow values to the delay tail.
    n_exp = length(Y_t)
    @assert length(m.flow) >= n_exp "flow must cover the expected series"
    flow_scale = 1.0 ./ m.flow[(end - n_exp + 1):end]   # load -> concentration

    # Observed series passes through unchanged (it is already a concentration).
    y = y_t isa NamedTuple ? y_t.y : y_t
    if y isa ComposableTuringIDModels.MissingObservations
        y_norm = y
    elseif ismissing(y)
        y_norm = Vector{Missing}(missing, n_exp)
    else
        @assert length(y) >= n_exp "observation series must be at least as long as the expected series"
        y_norm = y
    end

    # Divide the expected (load) series by flow to reach the concentration scale.
    Y_norm = Y_t .* flow_scale

    # Delegate scoring to the wrapped error model's generic loop, sampling it as
    # a submodel (the same seam `LatentDelay`/`Ascertainment` use) so the
    # returned `(; y_t, expected)` tuple carries the scored series rather than an
    # unevaluated model — required when this wrapper is composed inside a larger
    # model (e.g. an `IDModel` or `TransformObservationModel`).
    inner ~ as_turing_submodel(m.error_model, y_norm, Y_norm)
    return (; y_t = inner.y_t, expected = inner.expected)
end
end # module Sewage
