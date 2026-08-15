# Model front end: the README example as a composable `IDModel`.

"""
    model(; data = example_data(), distributions = example_distributions(),
        lpc_prior = Normal(log(2e11), 0.5),
        infection_model = Renewal(generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal()),
        observation_model = Ascertainment(
            LatentDelay(FlowNormalize(LogNormalError()), distributions.shedding_dist),
            lpc_prior)) -> IDModel

Assemble the wastewater model as a `ComposableTuringIDModels.IDModel`
(EpiSewer's README example). **Public but not exported** — call as
`EpiSewer.model(...)`. `infection_model` and `observation_model` are the
composable swap points; the defaults are a `Renewal` with random-walk `R_t`
and the per-case load → shedding delay → flow division → log-normal noise
chain. `data` and `distributions` parameterise those defaults; the daily flow
is data passed at `as_turing_model` time as `y_t = (y, flow)`.

# Arguments
- `data`, `distributions`: NamedTuples from `EpiSewer.example_data()` /
  `EpiSewer.example_distributions()`.
- `lpc_prior`: log-scale prior on the load shed per case (gc/case).
- `infection_model`, `observation_model`: the `ComposableTuringIDModels`
  components to compose; override either to swap that stage.

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
