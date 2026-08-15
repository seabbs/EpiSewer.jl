# Model front end: the README example as a composable `IDModel`.

"""
    model(; data = example_data(), distributions = example_distributions(),
        lpc_prior = Normal(log(2e11), 0.5),
        infection_model = Renewal(generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal()),
        observation_model = Ascertainment(
            LatentDelay(FlowNormalize(LogNormalError()), distributions.shedding_dist),
            lpc_prior)) -> IDModel

Build the wastewater model as a composable `ComposableTuringIDModels.IDModel`
(the EpiSewer README example). **Public but not exported** — call it as
`EpiSewer.model(...)`.

`infection_model` and `observation_model` are the composable interface; each
defaults to the README example assembly (a `Renewal` with random-walk `R_t`,
and an observation chain of per-case load scaling, shedding delay, flow
division, and relative log-normal noise). The daily flow is data, passed at
`as_turing_model` time as `y_t = (y = concentrations, flow = flow_vector)` —
never a `model()` argument.

# Arguments
- `data`: NamedTuple of DataFrames as returned by `EpiSewer.example_data()`.
- `distributions`: NamedTuple of discretised PMFs (`generation_dist`,
  `shedding_dist`, `incubation_dist`) as returned by
  `EpiSewer.example_distributions()`.
- `lpc_prior`: the log-scale prior on the load shed per case (gc/case), scaled
  onto expected infections by `Ascertainment`'s `xexpy` transform. Defaults to
  the scale implied by the Zurich example data.
- `infection_model`: the `AbstractInfectionModel` composable (defaults to the
  `Renewal` above).
- `observation_model`: the `AbstractObservationModel` composable (defaults to
  the flow-normalized chain above).

# Example
```@example model
using EpiSewer, ComposableTuringIDModels, Turing
import ComposableTuringIDModels: as_turing_model
d = EpiSewer.example_data()
y = d.measurements.concentration              # Union{Missing, Float64}, gc/mL
flow = Vector{Float64}(d.flows.flow)          # mL/day — data
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
        observation_model = Ascertainment(
            LatentDelay(
                FlowNormalize(LogNormalError()),
                distributions.shedding_dist,
            ),
            lpc_prior,
        ),
    )
    return IDModel(infection_model, observation_model)
end
