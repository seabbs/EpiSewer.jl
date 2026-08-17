# Model front end: the README example as a composable `IDModel`.

# EpiSewer's disease-specific distributions, as continuous distributions rather
# than PMFs: the components discretise them themselves (double-interval
# censoring, via `ComposableTuringIDModels`), so there is one discretisation
# path rather than two.
#
# The generation time is a **shifted** Gamma (`Gamma(...) + 1`, mean 3, sd 2.4),
# which is what R's `get_discrete_gamma_shifted` computes: its bins start at lag
# 1 (`.resources/EpiSewer/R/utils_dists.R`, `k <- 1:maxX`). Shifting puts no
# mass below 1, so `Renewal`'s drop-lag-0-and-renormalise step is exact.
const _GENERATION_TIME = Gamma(((3.0 - 1) / 2.4)^2, 2.4^2 / (3.0 - 1)) + 1
const _SHEDDING_DIST = Gamma(0.929639, 7.241397)
const _INCUBATION_DIST = Gamma(8.5, 0.4)

# Compose one delay onto an observation model. `LatentDelay`'s discretisation
# keywords exist only on its continuous-distribution constructor — a PMF vector
# is used as given, and an `UncertainDelay` (or any other prior model) carries
# its own horizon — so `D`/`Δd` are forwarded only where they are accepted.
# `nothing` means no convolution at all, which is the right default for the
# sewer residence time: EpiSewer's `residence_dist = c(1)`
# (`.resources/EpiSewer/R/EpiSewer.R`) is a point mass at same-day arrival, i.e.
# an identity convolution.
_delay(model, ::Nothing; D = nothing, Δd = 1.0) = model

function _delay(model, dist::ContinuousDistribution; D, Δd = 1.0)
    return LatentDelay(model, dist; D = D, Δd = Δd)
end

_delay(model, spec; D = nothing, Δd = 1.0) = LatentDelay(model, spec)

# The default observation chain. The **outermost** wrapper transforms the
# expected series first, so it is assembled innermost-first here and applies in
# the order Stan applies it
# (`.resources/EpiSewer/inst/stan/EpiSewer_main.stan`):
#
#   incubation  `lambda = convolve(inc_rev, I)`
#   load/case   per-case shed load, folded into `omega`
#   shedding    `omega_log = log_convolve(shedding_rev_log, ...)`
#   residence   `pi_log = log_convolve(residence_rev_log, omega_log)`
#   flow        `kappa_log = pi_log - flow_log`
function _observation_model(
        lpc_prior; shedding_dist, incubation_dist, residence_dist,
        D_shedding, D_incubation, D_residence, Δd
    )
    obs = FlowNormalize(LogNormalError())
    obs = _delay(obs, residence_dist; D = D_residence, Δd = Δd)
    obs = _delay(obs, shedding_dist; D = D_shedding, Δd = Δd)
    obs = Ascertainment(obs, lpc_prior)
    return _delay(obs, incubation_dist; D = D_incubation, Δd = Δd)
end

"""
    model(; generation_time, shedding_dist, incubation_dist,
        residence_dist = nothing, D_gen = 15.0, D_shedding = 38.0,
        D_incubation = 8.0, D_residence = nothing, Δd = 1.0,
        lpc_prior = Normal(log(2e11), 0.5),
        infection_model, observation_model) -> IDModel

Assemble the wastewater model as a `ComposableTuringIDModels.IDModel`
(EpiSewer's README example). **Public but not exported** — call as
`EpiSewer.model(...)`. `infection_model` and `observation_model` are the
composable swap points; the defaults are a `Renewal` with random-walk `R_t` and
the incubation delay → per-case load → shedding delay → flow division →
log-normal noise chain, with the incubation delay outermost so it acts on `I_t`
first (the shedding profile is indexed from symptom onset). The observed series
and the daily flow are both data, passed at `as_turing_model` time as
`y_t = (y = concentrations, flow = flow)`.

# Arguments
- `generation_time`, `shedding_dist`, `incubation_dist`, `residence_dist`: the
  delay inputs, forwarded untouched to `Renewal` and `LatentDelay`, each of
  which dispatches on what it is given. Any of them may be a continuous
  `Distribution` (discretised by double-interval censoring), an already
  discretised PMF vector, or a prior model such as an `UncertainDelay` (delay
  parameters carry priors, so the delay is inferred). `Renewal` drops the lag-0
  bin only when it discretises a distribution itself, so a *vector*
  `generation_time` is used verbatim and the caller owns the
  no-same-day-transmission convention. A `nothing` delay omits that
  convolution; `residence_dist` defaults to `nothing` because EpiSewer's
  `residence_dist = c(1)` is a point mass at same-day arrival, i.e. an identity
  convolution. The defaults are the EpiSewer README example's assumptions:
  a shifted `Gamma` generation time (mean 3, sd 2.4), `Gamma(0.929639,
  7.241397)` shedding load, and `Gamma(8.5, 0.4)` incubation period.
- `D_gen`, `D_shedding`, `D_incubation`, `D_residence`, `Δd`: right-truncation
  horizons and bin width used when a delay input is a continuous distribution.
  They match R's `maxX` values and are ignored for a PMF vector or a prior model
  (which carries its own horizon).
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
        generation_time = _GENERATION_TIME,
        shedding_dist = _SHEDDING_DIST,
        incubation_dist = _INCUBATION_DIST,
        residence_dist = nothing,
        D_gen = 15.0,
        D_shedding = 38.0,
        D_incubation = 8.0,
        D_residence = nothing,
        Δd = 1.0,
        lpc_prior = Normal(log(2.0e11), 0.5),
        infection_model = Renewal(
            ;
            generation_time = generation_time, rt = RandomWalk(),
            initialisation = Normal(), D_gen = D_gen, Δd = Δd,
        ),
        observation_model = _observation_model(
            lpc_prior;
            shedding_dist = shedding_dist,
            incubation_dist = incubation_dist,
            residence_dist = residence_dist,
            D_shedding = D_shedding,
            D_incubation = D_incubation,
            D_residence = D_residence,
            Δd = Δd,
        ),
    )
    return IDModel(infection_model, observation_model)
end
