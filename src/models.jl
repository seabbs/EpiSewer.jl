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
    Ascertainment, IDModel
using Distributions: Normal

const _FlowNormalize = Sewage.FlowNormalize

"""
    ww_idmodel(; data = example_data(), distributions = example_distributions(),
        flow = data.flows.flow, lpc_prior = Normal(log(1e9), 0.5))

Build the default wastewater model as a composable `IDModel` (the EpiSewer
README example).

The returned model can be evaluated with
`as_turing_model(ww_idmodel(), y_t, n)` where `y_t` is the observed
concentration series (length `n`, `missing` entries allowed) and `n` the
number of time points.

# Arguments
- `data`: NamedTuple of DataFrames (`measurements`, `flows`, `cases`) as
  returned by [`example_data`](@ref).
- `distributions`: NamedTuple of discretised PMFs (`generation_dist`,
  `shedding_dist`, `incubation_dist`) as returned by
  [`example_distributions`](@ref).
- `flow`: the daily flow series (mL/day), one value per time point. Defaults
  to the flows from `data`.
- `lpc_prior`: the log-scale prior on the load shed per case (scaled onto
  expected infections by `Ascertainment`'s default `xexpy` transform).

# Example
```julia
using EpiSewer, ComposableTuringIDModels, Turing
d = example_data()
y = [100.0, missing, 130.0]  # observed concentrations
mdl = as_turing_model(ww_idmodel(flow = d.flows.flow[1:3]), y, 3)
chn = sample(mdl, Prior(), 2)
```
"""
function ww_idmodel(;
        data = example_data(),
        distributions = example_distributions(),
        flow = Vector{Float64}(data.flows.flow),
        lpc_prior = Normal(log(1.0e9), 0.5),
    )
    gen = distributions.generation_dist
    shed = distributions.shedding_dist
    n = length(flow)

    # Core infection process: renewal with a random-walk R_t.
    infection_model = Renewal(
        ; generation_time = gen, rt = RandomWalk(), initialisation = Normal()
    )

    # Observation chain: infections -> load -> delayed load -> concentrations.
    obs_chain = _FlowNormalize(
        LatentDelay(
            Ascertainment(NormalError(), lpc_prior),
            shed,
        ),
        flow,
    )

    return IDModel(infection_model, obs_chain)
end
