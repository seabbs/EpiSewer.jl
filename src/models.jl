# Default wastewater model assembly (the EpiSewer README example expressed as
# a composable IDModel).
#
# The README model maps unobserved infections to observed flow-normalized
# concentrations through a chain of observation-model wrappers, all composed
# with ComposableTuringIDModels' `IDModel(infection_model, observation_model)`:
#
#   IDModel(
#     Renewal(generation_time, rt = RandomWalk(), initialisation = Normal()),
#     FlowNormalize(                    # observed concentrations (outermost)
#       LatentDelay(                    # shedding delay: convolve load with PMF
#         Ascertainment(                # load-per-case scaling
#           NormalError(),              # observation noise
#           Normal(log_lpc, σ_lpc)),
#         shedding_pmf),
#       flow))                          # flow normalization by daily flow
#
# The observation model receives `(y_t, I_t)` from the IDModel composite — the
# expected observations are the infection series — and each wrapper transforms
# them inward: Ascertainment multiplies by the per-case load, LatentDelay
# convolved-delays the load with the shedding-load PMF (truncating the expected
# series to the tail, which the inner error loop aligns against `y_t`), and
# FlowNormalize rescales both series by `flow ./ reference_flow` before the
# scores. This is the same chain EpiSewer's `sewer_job` builds (generation →
# shedding → sewage → observation), using only ecosystem + package components.

using ComposableTuringIDModels: Renewal, RandomWalk, NormalError, LatentDelay,
    Ascertainment, IDModel, TransformObservationModel
using Distributions: Normal

const _FlowNormalize = Sewage.FlowNormalize

"""
    model(; data = example_data(), distributions = example_distributions(),
        flow = data.flows.flow, lpc_prior = Normal(log(1e9), 0.5),
        infection_model = Renewal(generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal()),
        observation_model = FlowNormalize(
            LatentDelay(Ascertainment(NormalError(), lpc_prior),
                distributions.shedding_dist),
            flow)) -> IDModel

Build the wastewater model as a composable `ComposableTuringIDModels.IDModel`
(the EpiSewer README example). **Public but not exported** — call it as
`EpiSewer.model(...)`, never as a bare `model(...)`.

The two composable components — `infection_model` (an
`AbstractInfectionModel`) and `observation_model` (an
`AbstractObservationModel`) — are the model's interface. Each defaults to the
README example assembly:

  - `infection_model`: `Renewal(generation_time, rt = RandomWalk(),
    initialisation = Normal())` — a renewal process driven by a
    random-walk `R_t`.
  - `observation_model`:
    `FlowNormalize(LatentDelay(Ascertainment(NormalError(), lpc_prior),
    shedding_pmf), flow)` — load-per-case scaling, shedding delay, flow
    normalization, and a normal observation error.

Override either (or both) by passing your own `ComposableTuringIDModels`
component; the body is a single `IDModel(infection_model, observation_model)`
call. The data/prior arguments (`data`, `distributions`, `flow`,
`lpc_prior`) parameterize those defaults.

# Arguments
- `data`: NamedTuple of DataFrames (`measurements`, `flows`, `cases`) as
  returned by [`example_data`](@ref).
- `distributions`: NamedTuple of discretised PMFs (`generation_dist`,
  `shedding_dist`, `incubation_dist`) as returned by
  [`example_distributions`](@ref).
- `flow`: the daily flow series (mL/day), one value per time point. Defaults
  to the flows from `data`.
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
d = example_data()
# The series must be longer than the shedding-load PMF (~38 days, the
# LatentDelay lower bound) for the default observation chain.
y = fill(missing, 60)  # 60 days of (missing) concentrations
mdl = as_turing_model(EpiSewer.model(flow = Vector{Float64}(d.flows.flow[1:60])), y, 60)
chn = sample(mdl, Prior(), 2)
```
"""
function model(;
        data = example_data(),
        distributions = example_distributions(),
        flow = Vector{Float64}(data.flows.flow),
        lpc_prior = Normal(log(2.0e11), 0.5),
        infection_model = Renewal(
            ;
            generation_time = distributions.generation_dist,
            rt = RandomWalk(), initialisation = Normal(),
        ),
        # Observation chain: infections -> load -> delayed load -> concentrations.
        # The concentrations are scored on the LOG scale so the observation
        # noise is relative (a coefficient of variation, as in EpiSewer's
        # `noise_estimate`), rather than an absolute sigma that is negligible
        # on the ~1e2-3e3 gc/mL observed scale.
        observation_model = TransformObservationModel(
            _FlowNormalize(
                LatentDelay(
                    Ascertainment(NormalError(), lpc_prior),
                    distributions.shedding_dist,
                ),
                flow,
            ),
            x -> log.(x),
        ),
    )
    return IDModel(infection_model, observation_model)
end

# Backwards-compatible alias for the previous name of the default assembly.
ww_idmodel(args...; kwargs...) = model(args...; kwargs...)
