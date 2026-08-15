# Composable model front end: the README example as an `IDModel`.
#
# `EpiSewer.model()` returns `IDModel(infection_model, observation_model)`
# (ComposableTuringIDModels). Defaults assemble the EpiSewer README example:
# a `Renewal` infection process with random-walk `R_t`, and an observation
# chain `Ascertainment(LatentDelay(FlowNormalize(LogNormalError()), shed),
# lpc)` — per-case load scaling, shedding delay, flow division (gc/day ->
# gc/mL, via the thin `FlowNormalize` delegating to the ecosystem's
# `TransformObservationModel`), and relative (CV) log-normal noise.
#
# The daily flow is DATA, passed through the observation-data contract
# `y_t = (y = concentrations, flow = flow_vector)` at `as_turing_model` time.

const _FlowNormalize = Sewage.FlowNormalize

"""
    model(; data = example_data(), distributions = example_distributions(),
        lpc_prior = Normal(log(2e11), 0.5),
        infection_model = Renewal(generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal()),
        observation_model = Ascertainment(
            LatentDelay(
                FlowNormalize(LogNormalError()),
                distributions.shedding_dist),
            lpc_prior)) -> IDModel

Build the wastewater model as a composable `ComposableTuringIDModels.IDModel`
(the EpiSewer README example). **Public but not exported** — call it as
`EpiSewer.model(...)`, never as a bare `model(...)`.

`infection_model` and `observation_model` are the composable interface; each
defaults to the README example assembly. The data/prior arguments (`data`,
`distributions`, `lpc_prior`) parameterize those defaults. The flow is data:
pass it through the observation-data contract at `as_turing_model` time as
`y_t = (y = concentrations, flow = flow_vector)`, not as a `model()` argument.

# Arguments
- `data`: NamedTuple of DataFrames (`measurements`, `flows`, `cases`) as
  returned by `EpiSewer.example_data()`.
- `distributions`: NamedTuple of discretised PMFs (`generation_dist`,
  `shedding_dist`, `incubation_dist`) as returned by
  `EpiSewer.example_distributions()`.
- No `flow` argument: the daily flow series (mL/day) is data passed at
  `as_turing_model` time via `y_t = (y = concentrations, flow = flow_vector)`
  (see [`Sewage.FlowNormalize`](@ref)). Use the flows from `example_data()`
  (`d.flows.flow`) when reproducing the README example.
- `lpc_prior`: the log-scale prior on the load shed per case (gc/case),
  scaled onto expected infections by `Ascertainment`'s default `xexpy`
  transform. Defaults to the scale implied by the Zurich example data
  (`Normal(log(2e11), 0.5)`).
- `infection_model`: the `AbstractInfectionModel` composable to use as the
  infection process (defaults to the `Renewal` above).
- `observation_model`: the `AbstractObservationModel` composable to use as the
  observation chain (defaults to the flow-normalized chain above).

# Example
```julia
using EpiSewer, ComposableTuringIDModels, Turing
import ComposableTuringIDModels: as_turing_model
d = EpiSewer.example_data()
# The series must be longer than the shedding-load PMF (>=38 days, the
# LatentDelay lower bound); use the full 120-day series with missing kept.
y = d.measurements.concentration                       # Union{Missing, Float64}
flow = Vector{Float64}(d.flows.flow)                   # mL/day, data
mdl = as_turing_model(EpiSewer.model(), (y = y, flow = flow), length(y))
chn = sample(mdl, Prior(), 2)
```
"""
function model(;
        data = example_data(),
        distributions = example_distributions(),
        lpc_prior = Normal(log(2.0e11), 0.5),
        infection_model = Renewal(
            ;
            generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal(),
        ),
        # Observation chain: infections -> load -> delayed load -> concentrations.
        # Concentrations are scored with a LogNormal (relative, CV) noise model
        # so raw concentrations (gc/mL) can be passed straight to the model —
        # the flow-divided expected load is the LogNormal mean. This matches
        # EpiSewer's `noise_estimate` (a coefficient of variation), and avoids
        # an absolute sigma that would be negligible on the ~1e2-3e3 gc/mL
        # observed scale.
        observation_model = Ascertainment(
            LatentDelay(
                _FlowNormalize(Measurements.LogNormalError()),
                distributions.shedding_dist,
            ),
            lpc_prior,
        ),
    )
    return IDModel(infection_model, observation_model)
end
