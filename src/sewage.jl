"""
    EpiSewer.Sewage

Sewage-system components for the EpiSewer port: flow normalization of
concentration signals (`FlowNormalize`) and sewer residence-time
distributions.
"""
module Sewage

# Sewage-module components for the EpiSewer port.
#
# These are `ComposableTuringIDModels`-compatible structs: each concrete
# component is a subtype of an `AbstractComposableModel` role and implements
# the corresponding `as_turing_model` method, so it can be composed with the
# rest of the EpiAware ecosystem.

using Turing: Turing
using DynamicPPL: DynamicPPL, @model
using Distributions: Distributions
using ComposableTuringIDModels: ComposableTuringIDModels
import ComposableTuringIDModels: as_turing_model, as_turing_submodel
using ComposableTuringIDModels: AbstractObservationModel, TransformObservationModel

export FlowNormalize

"""
    FlowNormalize{E <: AbstractObservationModel}

A thin flow-normalization wrapper (the `flows_observe` component in EpiSewer).

EpiSewer models the pathogen **load** at the sampling site (gc/day) and derives
the observed concentration as `concentration_t = load_t / flow_t`. This wrapper
applies exactly that division to the expected series — which the model chain
provides as a load — so the scored expected values land on the observed
concentration scale (gc/mL).

The wrapper is deliberately thin: the division is delegated to the ecosystem's
[`TransformObservationModel`](@ref), whose transform maps the expected load
series to an expected concentration series (`Y -> Y ./ flow`). The **flow is
data**, passed through the observation-data contract at `as_turing_model` time
(as `y_t = (y = concentrations, flow = flow_vector)`), not stored on the model
— matching how [`BinomialError`](@ref) reads its trial counts from the data.
This keeps the wrapper a plain structure with no hand-rolled scoring loop.

Place `FlowNormalize` **inside** the chain, after the load-scaling and delay
components and before the observation-error model:

    Ascertainment(LatentDelay(FlowNormalize(LogNormalError(), ...), shed), lpc)

so the division is applied to the delayed load (not to infections).

# Fields
- `error_model`: the underlying observation model (e.g. `LogNormalError()`),
  operating on expected **concentrations**. Any `AbstractObservationModel`
  composes here.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
mn = FlowNormalize(NormalError())
# flow is data, passed at as_turing_model time:
model = as_turing_model(mn, (y = [100.0, missing], flow = [1.0e14, 1.5e14]), [100.0, 100.0])
```
"""
struct FlowNormalize{E <: AbstractObservationModel} <: AbstractObservationModel
    "The underlying observation model scoring expected concentrations."
    error_model::E
end

# The struct constructor `FlowNormalize(error_model)` is the only entry point;
# no convenience constructor is needed (there is nothing else to supply).

"""
    as_turing_model(m::FlowNormalize, y_t, Y_t)

Convert the expected **load** series (`Y_t`, gc/day) into an expected
**concentration** series by dividing by the daily flow, and delegate scoring
to the wrapped observation model.

The flow is read from the data contract: `y_t` must be a `NamedTuple`
`(y = observed_concentrations, flow = flow_vector)` (use `y = missing` to
simulate). The expected series is divided by the flow values aligned to its
tail — a `LatentDelay` above truncates the expected series from its right, so
the flow values corresponding to the retained trailing days are the ones used.
The observed `y` series passes through unchanged (it is already a
concentration).

The division is performed by composing the inner observation model with a
[`TransformObservationModel`](@ref) whose transform is `Y -> Y ./ flow_tail`
and sampling it as a submodel.
"""
@model function as_turing_model(m::FlowNormalize, y_t, Y_t)
    # Flow is data: required field of the data contract.
    hasflow = y_t isa NamedTuple && haskey(y_t, :flow)
    @assert hasflow "FlowNormalize needs y_t = (y = concentrations, flow = flow_vector)"

    y = y_t.y
    flow = y_t.flow
    n_exp = length(Y_t)
    @assert length(flow) >= n_exp "flow must cover the expected series"
    tail_flow = Vector{Float64}(flow[(end - n_exp + 1):end])

    # Divide the expected load series by flow: load -> concentration.
    composer = TransformObservationModel(m.error_model, Y -> Y ./ tail_flow)

    inner ~ as_turing_submodel(composer, y, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end

end # module Sewage
