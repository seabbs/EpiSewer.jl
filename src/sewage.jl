# Sewage components: flow normalization of concentration signals (`FlowNormalize`).

@doc raw"""
    FlowNormalize{E <: AbstractObservationModel}

Flow normalisation (the `flows_observe` component in EpiSewer): the expected
pathogen load arriving each day is divided by that day's wastewater flow to give
an expected concentration.

```math
\kappa_t = \frac{\pi_t}{q_t}
```

with ``\pi_t`` the expected load in gene copies per day, ``q_t`` the flow in
millilitres per day, and ``\kappa_t`` the expected concentration in gene copies
per millilitre. Dividing by the measured flow removes the day-to-day dilution
that rainfall and other flow variation introduce into the raw signal.

The division is performed by `TransformObservationModel`, and the **flow is
data**, passed at `as_turing_model` time as
`y_t = (y = concentrations, flow = flow)` rather than stored on the model — the
same data contract `BinomialError` uses for its trial counts.

When an outer `LatentDelay` has shortened the expected series, the flow is
aligned to its **tail**: the expected series always ends at the most recent day,
so the last ``\mathrm{length}(\pi)`` flow values are the matching ones.

Place `FlowNormalize` **inside** the chain, after the load-scaling and delay
components and before the observation-error model, so the division is applied
to the delayed load:

    Ascertainment(LatentDelay(FlowNormalize(LogNormalError(), ...), shed), lpc)

# Fields
- `error_model`: the underlying observation model, operating on expected
  concentrations.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
mn = EpiSewer.FlowNormalize(NormalError())
model = as_turing_model(mn, (y = [100.0, missing], flow = [1.0e14, 1.5e14]), [100.0, 100.0])
```
"""
struct FlowNormalize{E <: AbstractObservationModel} <: AbstractObservationModel
    "The underlying observation model scoring expected concentrations."
    error_model::E
end

"""
    as_turing_model(m::FlowNormalize, y_t, Y_t)

Divide the expected **load** series (`Y_t`, gc/day) by the daily flow to get
expected **concentrations**, and delegate scoring to the wrapped observation
model. `y_t` must be `(y = observed_concentrations, flow = flow_vector)`.

The flow values are aligned to the expected series' tail (a `LatentDelay`
above truncates the expected series from its right). The division is performed
by composing the inner model with a
`TransformObservationModel` whose transform is `Y -> Y ./ flow_tail`.
"""
@model function as_turing_model(m::FlowNormalize, y_t, Y_t)
    hasflow = y_t isa NamedTuple && haskey(y_t, :flow)
    @assert hasflow "FlowNormalize needs y_t = (y = concentrations, flow = flow_vector)"

    y = y_t.y
    flow = y_t.flow
    n_exp = length(Y_t)
    @assert length(flow) >= n_exp "flow must cover the expected series"
    tail_flow = Vector{Float64}(flow[(end - n_exp + 1):end])

    composer = TransformObservationModel(m.error_model, Y -> Y ./ tail_flow)

    inner ~ as_turing_submodel(composer, y, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end
