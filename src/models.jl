# Default wastewater model assembly (the EpiSewer README example expressed as
# a composable IDModel).
#
# The README model maps unobserved infections to observed flow-normalized
# concentrations through a chain of observation-model wrappers, all composed
# with ComposableTuringIDModels' `IDModel(infection_model, observation_model)`:
#
#   IDModel(
#     Infection model:
#       Renewal(generation_time, rt = RandomWalk(), initialisation = Normal()),
#     Observation chain (wrappers apply inward to the infection series I_t):
#       Ascertainment(                # load-per-case scaling: I_t * exp(lpc)
#         LatentDelay(                # shedding delay: convolve load with PMF
#           FlowNormalize(            # THIN: divide load by flow -> concentration
#             LogNormalError()),      # relative (CV) observation noise
#           shedding_pmf),
#         Normal(log_lpc, σ_lpc))
#
# The observation model receives `(y_t, I_t)` from the IDModel composite — the
# expected observations are the infection series — and each wrapper transforms
# them inward: Ascertainment multiplies by the per-case load, LatentDelay
# convolved-delays the load with the shedding-load PMF (truncating the expected
# series to the tail), and the thin FlowNormalize divides the delayed load by
# the daily flow (gc/day -> gc/mL) via the ecosystem's TransformObservationModel
# before the LogNormalError scores the raw concentrations with coefficient-of-
# variation noise. This is the same chain EpiSewer's `sewer_job` builds
# (generation -> shedding -> sewage -> observation), using only ecosystem +
# package components.
#
# NOTE: the daily flow is DATA, passed at `as_turing_model` time through the
# observation-data contract `y_t = (y = concentrations, flow = flow_vector)` —
# never stored on the model (see `Sewage.FlowNormalize`).

using ComposableTuringIDModels: Renewal, RandomWalk, LatentDelay,
    Ascertainment, IDModel
using Distributions: Normal

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

The two composable components — `infection_model` (an
`AbstractInfectionModel`) and `observation_model` (an
`AbstractObservationModel`) — are the model's interface. Each defaults to the
README example assembly:

  - `infection_model`: `Renewal(generation_time, rt = RandomWalk(),
    initialisation = Normal())` — a renewal process driven by a
    random-walk `R_t`.
  - `observation_model`:
    `Ascertainment(LatentDelay(FlowNormalize(LogNormalError()),
    shedding_pmf), lpc_prior)` — load-per-case scaling, shedding delay, flow
    normalization (load/flow -> concentration via the thin `FlowNormalize`),
    and a log-normal observation error (relative coefficient-of-variation
    noise). Raw concentrations (gc/mL) are passed directly as `y_t`.

The **flow is data**: pass it through the observation-data contract at
`as_turing_model` time as `y_t = (y = concentrations, flow = flow_vector)`,
not as a `model()` argument.

Override either (or both) by passing your own `ComposableTuringIDModels`
component; the body is a single `IDModel(infection_model, observation_model)`
call. The data/prior arguments (`data`, `distributions`, `lpc_prior`)
parameterize those defaults.

# Arguments
- `data`: NamedTuple of DataFrames (`measurements`, `flows`, `cases`) as
  returned by [`example_data`](@ref).
- `distributions`: NamedTuple of discretised PMFs (`generation_dist`,
  `shedding_dist`, `incubation_dist`) as returned by
  [`example_distributions`](@ref).
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
d = example_data()
# The series must be longer than the shedding-load PM (>=38 days, the
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

# Backwards-compatible alias for the previous name of the default assembly.
ww_idmodel(args...; kwargs...) = model(args...; kwargs...)
