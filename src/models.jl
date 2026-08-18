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
"""
    example_assumptions()

The disease assumptions the EpiSewer README uses, for the Zurich SARS-CoV-2
series returned by [`example_data`](@ref EpiSewer.example_data).

EpiSewer ships the same object as `ww_assumptions_SARS_CoV_2_Zurich`, and its
README builds it with `sewer_assumptions()`. Neither package treats these as
defaults: they are disease-specific inputs, so `model()` requires them and this
is one prepared set to pass.

Returns a `NamedTuple` ready to splat into [`model`](@ref EpiSewer.model):

- `generation_time`: shifted `Gamma`, mean 3, sd 2.4.
- `shedding_dist`: `Gamma(0.929639, 7.241397)`, relative to symptom onset.
- `incubation_dist`: `Gamma(8.5, 0.4)`, which puts the shedding profile on the
  infection timescale.
- `lpc_prior`: log-scale prior on the load shed per case. EpiSewer calibrates
  this from the case counts rather than assuming it (`load_per_case_calibrate()`
  is its default), and 1.1e11 gc/case is what its `suggest_load_per_case` returns
  for this series.
- `outlier_scale`: the concentration-scale size of one unit of outlier spike,
  R's `load_mean / flow_median`. That is 1.1e11 gc/case over this series' median
  flow of 1.545e11 mL/day.
- `initial_infections`: the crude seeding estimate, from
  [`crude_initial_infections`](@ref EpiSewer.crude_initial_infections) on the
  thinned series at that load per case. Matches R's `initial_cases_crude` to
  one decimal place.

# Examples
```@example example_assumptions
using EpiSewer
idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
nameof(typeof(idm))
```
"""
example_assumptions() = (
    generation_time = Gamma(((3.0 - 1) / 2.4)^2, 2.4^2 / (3.0 - 1)) + 1,
    shedding_dist = Gamma(0.929639, 7.241397),
    incubation_dist = Gamma(8.5, 0.4),
    lpc_prior = Normal(log(1.1e11), 0.5),
    initial_infections = 912.9,
    outlier_scale = 0.7119,
)

# EpiSewer's `R_estimate_gp()` priors, in days
# (`.resources/EpiSewer/R/model_infections.R`). It sums a short-term and a
# long-term Matérn-3/2 GP, so both terms are needed: the long-term one carries
# the larger magnitude, and dropping it would leave `R_t` far tighter than R's.
const _GP_LENGTH_SCALE_DAYS = 21.0          # length_scale_prior_mu = 7*3
const _GP_LENGTH_SCALE_SD_DAYS = 3.5        # length_scale_prior_sigma = 7/2
const _GP_MAGNITUDE = 0.125                 # magnitude_prior_mu
const _GP_MAGNITUDE_SD = 0.025              # magnitude_prior_sigma
const _GP_LONG_LENGTH_SCALE_DAYS = 84.0     # long_length_scale_prior_mu = 7*4*3
const _GP_LONG_LENGTH_SCALE_SD_DAYS = 7.0   # long_length_scale_prior_sigma
const _GP_LONG_MAGNITUDE = 0.25             # long_magnitude_prior_mu
const _GP_LONG_MAGNITUDE_SD = 0.05          # long_magnitude_prior_sigma

# EpiSewer's `R_t` link: `inv_softplus` with sharpness 4, applied to the summed
# Gaussian processes plus an intercept fixed at 1
# (`R/model_infections.R`: `R_link = c(0, 4, 0, 0)`, `R_intercept_prior_mu = 1`
# with `sigma = 0`; `inst/stan/functions/link.stan`: `apply_link`).
const _R_LINK_SHARPNESS = 4.0
const _R_INTERCEPT = 1.0

@doc raw"""
    softplus_link(z; intercept = 1.0, sharpness = 4.0)

EpiSewer's `inv_softplus` link from a latent path onto ``R_t``.

```math
R_t = \frac{\log\!\left(1 + e^{k (a + z_t)}\right)}{k}
```

with ``a`` = `intercept` and ``k`` = `sharpness`. At ``z_t = 0`` this gives
``R_t \approx 1``, and it is asymptotically **linear** in ``z_t`` rather than
exponential.

That distinction is the point of using it. `Renewal`'s default transformation is
`exp`, and ``\mathbb{E}[e^{z}] = e^{\sigma^2/2} > 1`` for a zero-mean ``z``, so an
`exp` link puts a systematic upward bias in ``\mathbb{E}[R_t]`` that compounds
through the renewal recursion. At the default Gaussian-process magnitudes that
bias is about 4% per generation, which over the example series is a factor of
several in the expected load. A softplus link is unbiased to first order, and it
is what R uses.

Returns the value on the **log** scale, because `Renewal` applies its own `exp`
to the latent path. Composing it as
`TransformLatentModel(gp, softplus_link)` therefore leaves `Renewal`'s
`transformation` free to keep seeding the initial incidence with `exp`, which is
what R does and what a single shared `transformation` could not express.

# Arguments
- `z`: the latent path, typically the summed Gaussian processes.
- `intercept`: the fixed intercept added before the link, R's `R_intercept`.
- `sharpness`: the softplus sharpness, R's second `R_link` element.

# Examples
```@example softplus_link
using EpiSewer
exp.(EpiSewer.softplus_link([-0.5, 0.0, 0.5]))
```

Against `exp`, which amplifies the upper tail:

```@example softplus_link
exp.([-0.5, 0.0, 0.5])
```
"""
function softplus_link(
        z; intercept = _R_INTERCEPT, sharpness = _R_LINK_SHARPNESS
    )
    return log.(_softplus.(intercept .+ z, sharpness))
end

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

# Arguments
- `days`: the length scale in days, as the literature and R state it.
- `n`: the length of the series the Gaussian process is generated over. This is
  the infection series length, so it includes the observation chain's lead-in
  (see [`observation_lead_in`](@ref EpiSewer.observation_lead_in)).
- `sd`: the standard deviation of the prior, in days. Given, the return value is
  a `Normal` truncated at the length-scale floor rather than a bare number.

# Examples
```@example gp_length_scale
using EpiSewer
EpiSewer.gp_length_scale(21.0, 164)
```

Twice the series length halves the standardised scale, because the standard
deviation of the index grows with it:

```@example gp_length_scale
EpiSewer.gp_length_scale(21.0, 328)
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

# Arguments
- `concentration`: the measured concentrations (gc/mL), in time order.
  `missing` entries are skipped.
- `flow`: the daily flow (mL/day), in time order.
- `load_per_case`: the load shed per case (gc/case), the median of
  `model()`'s `lpc_prior` on the natural scale.
- `days`: the length of the leading window averaged over. EpiSewer uses the
  first week.

# Examples
```@example crude_initial_infections
using EpiSewer
d = EpiSewer.example_data()
flow = Vector{Float64}(d.flows.flow)
EpiSewer.crude_initial_infections(d.measurements.concentration, flow, 2.0e11)
```

That is the scale the default seeding prior is centred on, against a series whose
measurements run from 145 to 3200 gc/mL.
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

# EpiSewer's `R_estimate_gp()`: the sum of a short-term and a long-term
# Matérn-3/2 Hilbert-space GP. Both are needed — the long-term magnitude (0.25)
# is twice the short-term one, so it carries most of the prior variability.
# `CombineLatentModels` sums them and prefixes each term's variables.
#
# The magnitudes and kernel transfer directly; the length scales do not, because
# R states them in days and `HilbertSpaceGP` standardises the index. `n` sets the
# series length that conversion assumes; see `gp_length_scale`.
function _default_rt(; n = 164, m = 20)
    short = HilbertSpaceGP(
        length_scale = gp_length_scale(
            _GP_LENGTH_SCALE_DAYS, n; sd = _GP_LENGTH_SCALE_SD_DAYS
        ),
        marginal_std = truncated(
            Normal(_GP_MAGNITUDE, _GP_MAGNITUDE_SD), 0.0, Inf
        ),
        m = m, c = 3.0, kernel = Matern32Kernel(),
    )
    long = HilbertSpaceGP(
        length_scale = gp_length_scale(
            _GP_LONG_LENGTH_SCALE_DAYS, n; sd = _GP_LONG_LENGTH_SCALE_SD_DAYS
        ),
        marginal_std = truncated(
            Normal(_GP_LONG_MAGNITUDE, _GP_LONG_MAGNITUDE_SD), 0.0, Inf
        ),
        m = m, c = 3.0, kernel = Matern32Kernel(),
    )
    combined = CombineLatentModels([short, long], ["gp_short", "gp_long"])
    # R's `inv_softplus` link, on the log scale so `Renewal`'s own `exp` recovers
    # `R_t`. Putting the link here rather than in `Renewal`'s `transformation` is
    # deliberate: that field is applied to the initial incidence as well as to the
    # `R_t` path, so a softplus there would also reshape the seeding prior, which
    # R keeps on the exponential scale.
    return TransformLatentModel(combined, softplus_link)
end

# `Renewal` discretises a continuous generation time in its keyword constructor
# but takes modifiers only in its positional one, which needs a discretised
# interval. Re-assembling from the deterministic model's own `gen_int` keeps one
# discretisation path (ComposableTuringIDModels#269).
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

# Compose one delay onto an observation model. `D`/`Δd` are forwarded only to
# the continuous-distribution constructor, the only one that accepts them.
# `nothing` omits the convolution, which is R's default for the sewer residence
# time (`residence_dist = c(1)`, a point mass at same-day arrival).
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

# The default observation chain, assembled innermost-first so the outermost
# wrapper transforms the expected series first, in Stan's order:
#
#   incubation  `lambda = convolve(inc_rev, I)`
#   load/case   per-case shed load, folded into `omega`
#   shedding    `omega_log = log_convolve(shedding_rev_log, ...)`
#   residence   `pi_log = log_convolve(residence_rev_log, omega_log)`
#   flow        `kappa_log = pi_log - flow_log`
function _observation_model(
        lpc_prior; shedding_dist, incubation_dist, residence_dist,
        D_shedding, D_incubation, D_residence, Δd, outlier_scale, load_cv
    )
    # Outliers sit immediately inside the flow division, so the spike lands on
    # the concentration scale where Stan adds it.
    inner = isnothing(outlier_scale) ? LogNormalError() :
        MeasurementOutliers(LogNormalError(); scale = outlier_scale)
    obs = FlowNormalize(inner)
    obs = _delay(obs, residence_dist; D = D_residence, Δd = Δd)
    obs = _delay(obs, shedding_dist; D = D_shedding, Δd = Δd)
    obs = Ascertainment(obs, lpc_prior)
    # Individual-level load variation applies to the shedding-onset series,
    # before the per-case load scaling, as `zeta_log` does in Stan.
    isnothing(load_cv) || (obs = LoadVariation(obs; cv = load_cv))
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
(EpiSewer's README example).

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
        generation_time,
        shedding_dist,
        incubation_dist,
        residence_dist = nothing,
        D_gen = 15.0,
        D_shedding = 38.0,
        D_incubation = 8.0,
        D_residence = nothing,
        Δd = 1.0,
        lpc_prior,
        n_gp = 164,
        rt = _default_rt(; n = n_gp),
        infection_noise = InfectionNoise(),
        initial_infections,
        outlier_scale = nothing,
        load_cv = 1.0,
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
            outlier_scale = outlier_scale,
            load_cv = load_cv,
        ),
    )
    return IDModel(infection_model, observation_model)
end
