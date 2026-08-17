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

# EpiSewer's `R_estimate_gp()` prior on the length scale, in days
# (`.resources/EpiSewer/R/model_infections.R`: `length_scale_prior_mu = 7*3`,
# `length_scale_prior_sigma = 7/2`).
const _GP_LENGTH_SCALE_DAYS = 21.0
const _GP_LENGTH_SCALE_SD_DAYS = 3.5

"""
    gp_length_scale(days, n; sd = nothing)

Convert a length scale in **days** to the units `HilbertSpaceGP` measures it in.

`HilbertSpaceGP` standardises the time index to zero mean and unit standard
deviation, so its length scale is in standard deviations of the index rather
than in days. A prior taken from the literature in days therefore has to be
divided by `std(1:n)` for a series of length `n`. Returns the scaled value, or a
truncated `Normal` prior when `sd` is given.

This is the conversion the package default uses to carry EpiSewer's
`length_scale_prior_mu = 21` days onto the standardised index.

# Examples
```jldoctest
julia> round(EpiSewer.gp_length_scale(21.0, 164); digits = 3)
0.442
```
"""
function gp_length_scale(days, n; sd = nothing)
    # Sample standard deviation of the integers `1:n`, in closed form:
    # `var(1:n) = n (n + 1) / 12`.
    s = sqrt(n * (n + 1) / 12)
    isnothing(sd) && return days / s
    return truncated(Normal(days / s, sd / s), 0.05, Inf)
end

# EpiSewer's seeding prior spread: `set_prior_normal_log(unit_factor = 10)` gives
# `sigma = log(10) / 2`, i.e. a factor of ten either side of the median at about
# two standard deviations (`.resources/EpiSewer/R/utils_modeldata.R`).
const _SEEDING_SPREAD = log(10) / 2

# The crude initial-infection estimate for the example series, from
# `crude_initial_infections` on the shipped data at the default load per case.
# Held as a constant so assembling a model reads no data.
const _INITIAL_INFECTIONS = 703.86

"""
    crude_initial_infections(concentration, flow, load_per_case; days = 7)

Crude estimate of the infections seeding a series, for the seeding prior.

Converts the mean measured concentration over the first `days` days into a case
count by scaling with the mean flow and dividing by the load shed per case,

```math
\\hat{I}_0 = 0.1 + \\frac{\\overline{c}\\; \\overline{q}}{\\text{load per case}}
```

with the small offset keeping the estimate positive when no pathogen was
measured. `missing` concentrations are skipped.

This is EpiSewer's `initial_cases_crude` (`R/sewer_modeldata.R`), which is what
its `seeding_estimate_rw()` centres the seeding prior on when no explicit
quantiles are given. Getting this scale right matters: the renewal process has
to reconcile the seeded infections with the measured concentrations, and a
seeding prior orders of magnitude away from the data can only be reconciled by a
sustained excursion in `R_t`.

Pair it with [`gp_length_scale`](@ref EpiSewer.gp_length_scale) to retune the
default model for a different series.

# Examples
```jldoctest
julia> d = EpiSewer.example_data();

julia> flow = Vector{Float64}(d.flows.flow);

julia> round(EpiSewer.crude_initial_infections(
           d.measurements.concentration, flow, 2.0e11); digits = 1)
703.9
```
"""
function crude_initial_infections(concentration, flow, load_per_case; days = 7)
    head = concentration[1:min(days, length(concentration))]
    observed = collect(skipmissing(head))
    isempty(observed) && throw(
        ArgumentError("no measured concentration in the first $days days")
    )
    c = sum(observed) / length(observed)
    q = flow[1:min(days, length(flow))]
    return 0.1 + c * (sum(q) / length(q)) / load_per_case
end

# EpiSewer's default `R_t` prior: a Hilbert-space approximate Gaussian process
# (`R_estimate_gp()`, R's `model_infections()` default). The scale-free settings
# transfer exactly — the Matérn 3/2 kernel (`matern_nu = c(3/2, ...)`), the
# boundary factor (`boundary_factor = 3`) and the magnitude prior
# (`magnitude_prior_mu = 0.125`, `magnitude_prior_sigma = 0.025`), which is in
# log-`R_t` units either way.
#
# The length scale does not: R states it in days, and `HilbertSpaceGP`
# standardises the index, so it depends on the series length. `n` is not known
# when the model is assembled, so the default is R's 21 ± 3.5 days converted at
# the length of the example series. `gp_length_scale` does the conversion for a
# materially different series length.
function _default_rt(; n = 164)
    return HilbertSpaceGP(
        length_scale = gp_length_scale(
            _GP_LENGTH_SCALE_DAYS, n; sd = _GP_LENGTH_SCALE_SD_DAYS
        ),
        marginal_std = truncated(Normal(0.125, 0.025), 0.0, Inf),
        m = 20, c = 3.0, kernel = Matern32Kernel(),
    )
end

# Compose one delay onto an observation model. `LatentDelay`'s discretisation
# keywords exist only on its continuous-distribution constructor — a PMF vector
# is used as given, and an `UncertainDelay` (or any other prior model) carries
# its own horizon — so `D`/`Δd` are forwarded only where they are accepted.
# `nothing` means no convolution at all, which is the right default for the
# sewer residence time: EpiSewer's `residence_dist = c(1)`
# (`.resources/EpiSewer/R/EpiSewer.R`) is a point mass at same-day arrival, i.e.
# an identity convolution.
# Assemble the infection process: a `Renewal` with `rt` as its `R_t` prior, and
# the stochastic-infection modifier composed onto its step.
#
# `Renewal` discretises a continuous generation time in its keyword constructor
# and takes modifiers only in its positional one, which needs an already
# discretised interval. Building the deterministic model first and re-assembling
# it from its own `gen_int` keeps one discretisation path rather than
# discretising again here (ComposableTuringIDModels issue #269).
function _infection_model(generation_time, rt, noise, initialisation; D_gen, Δd)
    base = Renewal(;
        generation_time = generation_time, rt = rt,
        initialisation = initialisation, D_gen = D_gen, Δd = Δd
    )
    isnothing(noise) && return base
    base.gen_int isa AbstractVector{<:Real} || throw(
        ArgumentError(
            "an inferred generation time cannot carry an infection-noise " *
                "modifier: pass `infection_noise = nothing`, or give " *
                "`infection_model` directly"
        )
    )
    return Renewal(
        base.gen_int, noise; rt = rt, initialisation = initialisation
    )
end

_delay(model, ::Nothing; D = nothing, Δd = 1.0) = model

function _delay(model, dist::ContinuousDistribution; D, Δd = 1.0)
    return LatentDelay(model, dist; D = D, Δd = Δd)
end

_delay(model, spec; D = nothing, Δd = 1.0) = LatentDelay(model, spec)

# Lead-in contributed by one component and everything nested inside it. Only a
# `LatentDelay` contributes; every other wrapper passes through the lead-in of
# the model it wraps, whatever that child field is called (`model` on the
# ecosystem's modifiers, `error_model` on `FlowNormalize`), so the walk covers
# any chain rather than the default one only.
_lead_in(::Any) = 0

_lead_in(d::LatentDelay) = _delay_length(d.delay) - 1 + _lead_in(d.model)

function _lead_in(obs::AbstractObservationModel)
    return sum(
        _lead_in(getfield(obs, f)) for f in fieldnames(typeof(obs)); init = 0
    )
end

# Parallel streams (a `Split`'s `streams`): each branch scores its own
# observations, so the series must be long enough for the longest branch.
function _lead_in(branches::Union{Tuple, NamedTuple, AbstractVector})
    models = filter(x -> x isa AbstractObservationModel, collect(branches))
    return isempty(models) ? 0 : maximum(_lead_in, models)
end

# Delay PMF length, for each shape `LatentDelay` stores its delay in: a fixed
# PMF (held reversed), a per-time sequence of PMFs (all the same length), or an
# `UncertainDelay`, whose PMF length is held constant across draws by its
# horizon `D` and bin width `Δd`.
_delay_length(pmf::AbstractVector{<:Real}) = length(pmf)
_delay_length(pmfs::AbstractVector{<:AbstractVector{<:Real}}) =
    length(first(pmfs))
_delay_length(u::UncertainDelay) = length(0.0:u.Δd:(u.D - u.Δd))

function _delay_length(delay)
    return error(
        "Cannot determine the delay PMF length of a $(typeof(delay)); " *
            "`EpiSewer.observation_lead_in` knows the fixed, time-varying and " *
            "`UncertainDelay` shapes."
    )
end

"""
    observation_lead_in(model) -> Int

Extra leading time points the **infection** series needs so that every
observation is scored.

Each `LatentDelay` in the chain shortens the expected series by
`length(pmf) - 1`, dropping the partially observed head of the convolution, and
the observation-error model then right-aligns the observations against what is
left. A caller who passes `n = length(y)` to `as_turing_model` therefore leaves
the first `observation_lead_in(model)` observations silently unscored. Pass

    n = length(y) + EpiSewer.observation_lead_in(mdl)

instead: the expected series comes out at `length(y)` and every observation is
scored. This is the composable equivalent of the R model's `L + S + D` lead-in.

# Arguments
- `model`: an `IDModel` (its observation chain is walked) or a bare observation
  model. The walk sums over the nested `LatentDelay`s — fixed PMF, per-time PMF
  sequence, or an `UncertainDelay` — so the lead-in is `0` for a chain with no
  delay, grows if a residence delay is added, and takes the longest branch of a
  chain that splits into parallel streams.

# Example
The `n` to pass for a given observed series, with the default chain:

```@example observation_lead_in
using EpiSewer
mdl = EpiSewer.model()
y = EpiSewer.example_data().measurements.concentration
lead_in = EpiSewer.observation_lead_in(mdl)
(observations = length(y), lead_in = lead_in, n = length(y) + lead_in)
```
"""
observation_lead_in(mdl::IDModel) = _lead_in(mdl.observation_model)

observation_lead_in(obs::AbstractObservationModel) = _lead_in(obs)

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
        lpc_prior = Normal(log(2e11), 0.5), n_gp = 164, rt,
        infection_noise = InfectionNoise(), initial_infections = 703.86,
        seeding, infection_model, observation_model) -> IDModel

Assemble the wastewater model as a `ComposableTuringIDModels.IDModel`
(EpiSewer's README example). **Public but not exported** — call as
`EpiSewer.model(...)`.

The defaults are EpiSewer's default model: a `Renewal` whose `R_t` follows a
Hilbert-space approximate Gaussian process, with stochastic infections and a
data-scaled seeding prior, observed through the incubation delay → per-case load
→ shedding delay → flow division → log-normal noise chain. The incubation delay
is outermost so it acts on `I_t` first, because the shedding profile is indexed
from symptom onset. `infection_model` and `observation_model` are the composable
swap points. The observed series and the daily flow are both data, passed at
`as_turing_model` time as `y_t = (y = concentrations, flow = flow)`.

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
- `rt`: the `R_t` prior. Defaults to EpiSewer's `R_estimate_gp()`: a
  `HilbertSpaceGP` with a Matérn 3/2 kernel, boundary factor 3, and a
  `Normal(0.125, 0.025)` magnitude prior, all of which carry over directly. Its
  length scale does not, because R states it in days while `HilbertSpaceGP`
  measures it in standard deviations of the time index. `n_gp` sets the series
  length that conversion assumes, so the default is R's 21 ± 3.5 days at the
  length of the example series; see
  [`gp_length_scale`](@ref EpiSewer.gp_length_scale) for another series.
- `infection_noise`: the stochastic-infection modifier
  ([`InfectionNoise`](@ref EpiSewer.InfectionNoise)), EpiSewer's
  `infection_noise_estimate()`. Pass `nothing` for a deterministic renewal
  process.
- `initial_infections`, `seeding`: the seeding prior, EpiSewer's
  `seeding_estimate_rw()` intercept. `seeding` is a prior on **log** initial
  infections, defaulting to a `Normal` centred on `log(initial_infections)` with
  R's factor-of-ten spread. The default `initial_infections` is the crude
  estimate for the example series; compute it for another series with
  [`crude_initial_infections`](@ref EpiSewer.crude_initial_infections). This
  scale is load bearing rather than cosmetic: the renewal process has to
  reconcile the seeded infections with the measured concentrations, so a seeding
  prior orders of magnitude from the data forces a sustained `R_t` excursion to
  compensate.
- `infection_model`, `observation_model`: the `ComposableTuringIDModels`
  components to compose; override either to swap that stage. Giving
  `infection_model` directly supersedes `rt`, `infection_noise` and `seeding`.

# Example
```@example model
using EpiSewer, ComposableTuringIDModels, Turing
import ComposableTuringIDModels: as_turing_model
d = EpiSewer.example_data()
y = d.measurements.concentration              # Union{Missing, Float64}, gc/mL
flow = Vector{Float64}(d.flows.flow)          # mL/day — data
idm = EpiSewer.model()
# The infection series needs the observation chain's lead-in on top of the
# observed days, so that every observation is scored (`observation_lead_in`).
n = length(y) + EpiSewer.observation_lead_in(idm)
mdl = as_turing_model(idm, (y = y, flow = flow), n)
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
        n_gp = 164,
        rt = _default_rt(; n = n_gp),
        infection_noise = InfectionNoise(),
        initial_infections = _INITIAL_INFECTIONS,
        seeding = Normal(log(initial_infections), _SEEDING_SPREAD),
        infection_model = _infection_model(
            generation_time, rt, infection_noise, seeding;
            D_gen = D_gen, Δd = Δd
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
