# Infection components: stochastic infections (`infection_noise_estimate`) and
# the seeding-phase random walk (`seeding_estimate_rw`).

# Map a standard normal onto the family whose moments are the negative
# binomial's. `reparameterise` is ReparameterisedDistributions' caller-facing
# verb; `check_args = false` scores an invalid moment pair `-Inf` rather than
# throwing mid-gradient. The two location-scale families are written out because
# the generic path allocates a distribution per step inside the differentiated
# scan and loses the tails through a `cdf`/`quantile` round trip.
function _draw(::Type{LogNormal}, mean, sd, z)
    σ² = log1p((sd / mean)^2)
    return exp(log(mean) - σ² / 2 + z * sqrt(σ²))
end
_draw(::Type{Normal}, mean, sd, z) = mean + sd * z
function _draw(family, mean, sd, z)
    d = reparameterise(family; mean = mean, sd = sd, check_args = false)
    return quantile(d, cdf(Normal(), z))
end

# The field type a noise family is stored under. `typeof(LogNormal)` is
# `UnionAll`, which says nothing about *which* family it is, so `_draw`'s
# dispatch is resolved at run time on every scan step and the incidence it
# returns infers as `Any` — which then propagates through the whole renewal
# recursion. `Type{LogNormal}` has one instance, so the family stays in the
# type domain and the scan stays concretely typed.
_family_type(::Type{F}) where {F} = Type{F}
_family_type(dist) = typeof(dist)

@doc raw"""
    InfectionNoise{D, X, C} <: AbstractRenewalModifier

Stochastic infections for a renewal process (the `infection_noise_estimate`
component in EpiSewer).

A deterministic renewal process fixes infections at the renewal expectation
``\iota_t``. This modifier gives them their own noise, so infections carry
variance beyond what the reproduction number implies. The machinery is generic
and `dist` is the family it is applied to: the mean and variance are matched to a
negative binomial, and the draw is non-centred.

```math
\mathbb{E}[I_t] = \iota_t, \qquad
\mathrm{Var}[I_t] = \iota_t \left(1 + \iota_t \xi^2\right), \qquad
I_t = F^{-1}_{\iota_t, c_t \iota_t}\!\left(\Phi(\tilde{I}_t)\right),
\qquad \tilde{I}_t \sim \mathrm{Normal}(0, 1)
```

where ``\iota_t = R_t \sum_{i} I_{t-i} g_i`` is the renewal expectation, ``\xi``
is the overdispersion, and ``F^{-1}_{m, s}`` is the quantile function of `dist`
matched to mean ``m`` and standard deviation ``s``. Those are the moments of a
negative binomial with mean ``\iota_t`` and overdispersion ``\xi``, so the draw
is an approximation to one; ``\xi = 0`` leaves Poisson variance. The moment solve
is `ReparameterisedDistributions.reparameterise`, so any family it supports for
`(:mean, :sd)` can be passed.

For the two location-scale families the transformation is exact and closed form,
which is what makes ``\tilde{I}_t`` a plain standard normal rather than a
round-trip through ``\Phi``:

```math
\sigma_t^2 = \log\!\left(1 + c_t^2\right), \qquad
\log I_t = \log \iota_t - \frac{\sigma_t^2}{2} + \tilde{I}_t \sigma_t
\qquad (\texttt{LogNormal})
```

```math
I_t = \iota_t + \tilde{I}_t \, c_t \iota_t \qquad (\texttt{Normal})
```

The default `LogNormal` keeps infections positive by construction, so no
truncation is needed. `Normal` is R's own choice, but R bounds it
(`I ~ normal(iota, ...) T[0, ]`) and this modifier cannot, so it can produce a
negative expected concentration.

``c_t`` is the coefficient of variation ``\sqrt{1/\iota_t + \xi^2}`` under a
smooth upper limit ``u - \mathrm{softplus}(u - c, k)``, which holds it finite as
``\iota_t`` approaches zero. R applies the same limit with the same constants.
`cv_cap = Inf` removes it and restores the exact variance.

The parameterisation is **non-centred**: the sampled quantity is the standard
normal ``\tilde{I}_t`` and the location and scale are applied afterwards. This is
forced rather than chosen: `apply_modifier` is deterministic and a modifier's
priors resolve before the scan, so a modifier cannot draw ``I_t`` conditional on
``\iota_t``.

Because the modified incidence is what the renewal scan feeds forward, the noise
compounds through the process rather than perturbing each day in isolation.

# Fields
- `dist`: the noise family, matched to the negative-binomial moments. Defaults to
  `LogNormal`. `Normal` gives R's own linear form. Any family
  `ReparameterisedDistributions` registers for `(:mean, :sd)` is accepted, and
  the closed-form transformation covers `Normal` and `LogNormal`.
- `overdispersion`: the prior for ``\xi``, or a fixed scalar. EpiSewer fixes it
  at 0.1 (`overdispersion_prior_sigma = 0`), which is the default here.
- `cv_cap`, `cv_sharpness`: the soft limit on the coefficient of variation, R's
  `0.5` and `10`. An infinite `cv_cap` removes the limit and restores the exact
  negative-binomial variance, at the cost of the funnel.

# Example
```julia
using EpiSewer, ComposableTuringIDModels, Distributions
r = Renewal(
    [0.2, 0.3, 0.5], EpiSewer.InfectionNoise();
    rt = RandomWalk(), initialisation = Normal()
)
# R's linear form instead of the positive default.
noise = EpiSewer.InfectionNoise(; dist = Normal)
```
"""
struct InfectionNoise{D, X, C} <: AbstractRenewalModifier
    "The noise family, matched to the negative-binomial moments."
    dist::D
    "Prior for the overdispersion ``\\xi``, or a fixed scalar."
    overdispersion::X
    "Soft upper limit on the coefficient of variation."
    cv_cap::C
    "Sharpness of the soft limit."
    cv_sharpness::C

    # Inner constructor so that no default one is generated: the family has to
    # reach the field as `Type{F}` rather than as `UnionAll` (see
    # `_family_type`).
    function InfectionNoise(
            dist, overdispersion, cv_cap::C, cv_sharpness::C
        ) where {C}
        return new{_family_type(dist), typeof(overdispersion), C}(
            dist, overdispersion, cv_cap, cv_sharpness
        )
    end
end

function InfectionNoise(;
        dist = LogNormal, overdispersion = 0.1, cv_cap = 0.5,
        cv_sharpness = 10.0
    )
    cap, sharp = promote(cv_cap, cv_sharpness)
    return InfectionNoise(dist, overdispersion, cap, sharp)
end

@doc raw"""
    InfectionNoiseDraws{V, D, X, C} <: AbstractRenewalModifier

A resolved [`InfectionNoise`](@ref): the drawn standard normals, ready to scan.

This is what [`InfectionNoise`](@ref) returns from its pre-scan
`as_turing_model` seam, following the `ImportedCases` → `ImportedRate` pattern.
Its substate is the step counter, so step ``t`` reads its own ``\tilde{I}_t``.

# Fields
- `raw`: the standard normal draws ``\tilde{I}_t``, one per time.
- `dist`: the noise family.
- `overdispersion`: the resolved ``\xi``.
- `cv_cap`, `cv_sharpness`: the soft limit on the coefficient of variation.
"""
struct InfectionNoiseDraws{V, D, X, C} <: AbstractRenewalModifier
    "The standard normal draws, one per time."
    raw::V
    "The noise family."
    dist::D
    "The resolved overdispersion."
    overdispersion::X
    "Soft upper limit on the coefficient of variation."
    cv_cap::C
    "Sharpness of the soft limit."
    cv_sharpness::C

    # As on [`InfectionNoise`](@ref): the family reaches the field as
    # `Type{F}`, so `apply_modifier`'s `_draw` call resolves at compile time.
    function InfectionNoiseDraws(
            raw, dist, overdispersion, cv_cap::C, cv_sharpness::C
        ) where {C}
        return new{typeof(raw), _family_type(dist), typeof(overdispersion), C}(
            raw, dist, overdispersion, cv_cap, cv_sharpness
        )
    end
end

# The substate is the step counter: a scan step has no clock of its own, so a
# per-time modifier carries the index it has reached. The window is unused.
modifier_init_state(::InfectionNoiseDraws, window) = 0

# R's `soft_upper(x, u, k) = u - softplus(u - x, k)`
# (`inst/stan/functions/link.stan`). Written in the stable form because
# `log1p(exp(k x)) / k` overflows once `k x` passes about 709, which a diverging
# proposal reaches. An infinite limit is no limit, so it returns `x`.
_softplus(x, k) = (k * x > 0 ? x : zero(x)) + log1p(exp(-abs(k * x))) / k
_soft_upper(x, u, k) = isfinite(u) ? u - _softplus(u - x, k) : x

function apply_modifier(mod::InfectionNoiseDraws, incidence, t)
    ξ = mod.overdispersion
    ι = incidence
    # The negative binomial's coefficient of variation,
    # `sqrt(iota (1 + iota ξ²)) / iota`, simplified.
    cv = _soft_upper(sqrt(inv(ι) + ξ^2), mod.cv_cap, mod.cv_sharpness)
    return _draw(mod.dist, ι, cv * ι, mod.raw[t + 1]), t + 1
end

@doc raw"""
    as_turing_model(mod::InfectionNoise, n)

Sample the non-centred infection noise ahead of the scan.

Draws `n` standard normals through the `IID` seam and resolves the
overdispersion slot, returning the [`InfectionNoiseDraws`](@ref) the scan uses.
A fixed scalar `overdispersion` is passed straight through, so EpiSewer's fixed
``\xi = 0.1`` costs no parameter.

# Arguments
- `mod`: the [`InfectionNoise`](@ref) modifier.
- `n`: the length of the renewal series.
"""
@model function as_turing_model(mod::InfectionNoise{<:Any, <:Real}, n)
    # A fixed overdispersion is a constant, not a point-mass parameter: that is
    # what `infection_noise_estimate(overdispersion_prior_sigma = 0)` means, and
    # it keeps `ξ` out of the sampled space entirely.
    I_raw ~ as_turing_submodel(IID(Normal()), n; prefix = true)
    return InfectionNoiseDraws(
        I_raw, mod.dist, mod.overdispersion, mod.cv_cap, mod.cv_sharpness
    )
end

@model function as_turing_model(mod::InfectionNoise, n)
    I_raw ~ as_turing_submodel(IID(Normal()), n; prefix = true)
    ξ ~ as_turing_submodel(mod.overdispersion, n; prefix = true)
    return InfectionNoiseDraws(
        I_raw, mod.dist, ξ, mod.cv_cap, mod.cv_sharpness
    )
end

# --- Seeding: a random walk over the seeding window -------------------------
#
# The seeding window is the incidence the renewal recursion starts from, one
# value per generation-interval lag. `ConstantRenewalStep` fills it with a
# deterministic exponential at the growth rate implied by `R₀`; R's
# `seeding_estimate_rw` instead gives it its own geometric random walk. The
# window is built by `renewal_init_window`, which is a method on the renewal
# *core*, so the walk is a core rather than a modifier: a modifier is handed the
# window after it has been built and cannot change it.

# The number of innovations the walk needs: one fewer than the window, because
# the walk starts *at* the intercept. Taken off the generation interval, since
# that is what fixes the window's length — the pre-scan seam is handed the
# series length, which is a different number.
_n_seed_innovations(g::AbstractVector) = length(g) - 1

function _n_seed_innovations(::Nothing)
    return throw(
        ArgumentError(
            "this `SeedingRandomWalk` carries no generation interval, so the " *
                "length of its seeding window is unknown. `EpiSewer.model()` " *
                "fills the interval in from `generation_time`; to build " *
                "one by hand, pass the reversed interval and the mixing " *
                "operator to the three-argument constructor."
        )
    )
end

# The walk itself: a cumulative sum of the innovations, anchored at zero, so the
# path has one more element than it has innovations and its FIRST element is the
# anchor. `vcat` of a scalar onto `cumsum` rather than a mutated buffer, so the
# whole path stays differentiable.
_seed_walk(innovations) = vcat(zero(eltype(innovations)), cumsum(innovations))

@doc raw"""
    SeedingRandomWalk{T, K, S} <: AbstractConstantRenewalStep

A geometric random walk over the seeding window (EpiSewer's
`seeding_estimate_rw`).

The seeding window is the `len_gen_int` incidences the renewal recursion starts
from, before it can be applied. `ComposableTuringIDModels`' default renewal core
fills that window with a deterministic exponential at the growth rate implied by
``R_0``. This core replaces it with a random walk on log infections,

```math
\log \iota_1 = \log I_0, \qquad
\log \iota_{t} = \log \iota_{t-1} + \sigma \tilde{\epsilon}_t, \qquad
\tilde{\epsilon}_t \sim \mathrm{Normal}(0, 1)
```

for ``t = 2, \dots, `` `len_gen_int`, which is R's
`iota[1:(G+se)] = exp(random_walk([iota_log_seed_intercept]',
iota_log_ar_noise, 0))`.
The parameterisation is **non-centred**: the sampled quantity is the standard
normal ``\tilde{\epsilon}_t`` and the step size is applied afterwards, which is
what R's `vector<multiplier=iota_log_seed_sd[1]>` declaration does.

The growth rate implied by ``R_0`` is **not** used. The walk determines the
shape of the window, so the seeding phase no longer has to be an exponential at
the initial reproduction number. R does the same. Nothing in its seeding block
reads `R[1]`.

!!! note "`initial_infections` anchors the *earliest* seeded day"
    This changes what `model()`'s `seeding` prior (and so `initial_infections`)
    refers to. The window runs oldest to newest, and the default deterministic
    seeding puts ``I_0`` at its **newest** entry, decaying backwards from there.
    R's intercept is `iota[1]`, the **earliest** entry, and its
    `initial_cases_crude` is documented as the cases at the *start* of the time
    series. This core follows R. With all innovations at zero the window is
    constant at ``I_0``, and with non-zero innovations ``I_0`` is its first
    element.

The walk spans the generation interval. R's seeding phase is `G + se` days,
where `se` extends it when the series opens with a run of non-detects longer
than the generation interval. The two agree whenever `se = 0`. That is every
series reaching three consecutive detects within its first `G` days, the
shipped example among them.

Single-series only. A stratified renewal would need one walk per stratum, which
this does not provide.

# Fields
- `rev_gen_int`: the reversed generation interval, which fixes the window's
  length. `nothing` on an instance built by the keyword constructor, which is a
  specification `model()` completes once it has discretised the generation time.
- `mixing`: the coupling operator, as on any renewal core. `nothing` alongside a
  `nothing` interval.
- `step_size`: the prior for ``\sigma``, or a fixed scalar. Defaults to R's
  `truncated(Normal(0.05, 0.025), 0, Inf)`, from `rel_change_prior_mu = 0.05`
  and `rel_change_prior_sigma = 0.025`, about a 5% relative change per day. A
  fixed scalar costs no parameter.

# Example
```jldoctest
julia> using EpiSewer

julia> idm = EpiSewer.model(;
           EpiSewer.example_assumptions()...,
           seeding_walk = EpiSewer.SeedingRandomWalk(),
       );

julia> nameof(typeof(idm.infection_model.recurrent_step.core))
:SeedingRandomWalk
```
"""
struct SeedingRandomWalk{T, K, S} <: AbstractConstantRenewalStep
    "The reversed generation interval, or `nothing` on a specification."
    rev_gen_int::T
    "The coupling operator, or `nothing` on a specification."
    mixing::K
    "Prior for the walk's step size ``\\sigma``, or a fixed scalar."
    step_size::S
end

# R's `rel_change_prior_mu` / `rel_change_prior_sigma`
# (`.resources/EpiSewer/R/model_infections.R`, `seeding_estimate_rw`), scored as
# `iota_log_seed_sd[1] ~ normal(...) T[0, ]` in `EpiSewer_main.stan`.
const _SEEDING_STEP_PRIOR = truncated(Normal(0.05, 0.025), 0.0, Inf)

function SeedingRandomWalk(; step_size = _SEEDING_STEP_PRIOR)
    return SeedingRandomWalk(nothing, nothing, step_size)
end

# Complete a specification from the renewal core `Renewal` built, so the walk
# inherits exactly the interval and mixing the rest of the model uses.
function _seed_with(walk::SeedingRandomWalk, core::AbstractConstantRenewalStep)
    return SeedingRandomWalk(core.rev_gen_int, core.mixing, walk.step_size)
end

@doc raw"""
    SeedingRandomWalkDraws{T, K, V} <: AbstractConstantRenewalStep

A resolved [`SeedingRandomWalk`](@ref): the drawn walk, ready to scan.

This is what [`SeedingRandomWalk`](@ref) returns from its pre-scan
`as_turing_model` seam, following the [`InfectionNoise`](@ref) →
[`InfectionNoiseDraws`](@ref) pattern. The innovations are stored already scaled
by the step size, so the seeding window is `I₀ * exp(cumsum)` with the anchor
prepended.

# Fields
- `rev_gen_int`: the reversed generation interval.
- `mixing`: the coupling operator.
- `innovations`: the scaled walk innovations ``\sigma \tilde{\epsilon}_t``, one
  fewer than the seeding window has entries.
"""
struct SeedingRandomWalkDraws{T, K, V} <: AbstractConstantRenewalStep
    "The reversed generation interval."
    rev_gen_int::T
    "The coupling operator."
    mixing::K
    "The scaled walk innovations, one fewer than the window has entries."
    innovations::V
end

@doc raw"""
    renewal_init_window(step::SeedingRandomWalkDraws, I₀, r, len_gen_int)

The seeding window as a geometric random walk anchored at `I₀`.

Returns `len_gen_int` strictly positive values, oldest first, whose **first**
entry is `I₀`. `r`, the growth rate implied by ``R_0``, is ignored: the walk
replaces the exponential the default core would build (see
[`SeedingRandomWalk`](@ref)).

# Arguments
- `step`: the resolved seeding walk.
- `I₀`: the initial incidence, the walk's anchor.
- `r`: the growth rate implied by ``R_0``, unused.
- `len_gen_int`: the number of lags the generation interval covers.
"""
function renewal_init_window(
        step::SeedingRandomWalkDraws, I₀::Real, r, len_gen_int
    )
    length(step.innovations) + 1 == len_gen_int || throw(
        DimensionMismatch(
            "a seeding walk with $(length(step.innovations)) innovations " *
                "spans $(length(step.innovations) + 1) days, but the " *
                "generation interval covers $len_gen_int lags"
        )
    )
    return I₀ .* exp.(_seed_walk(step.innovations))
end

# A stratified renewal would need one walk per stratum; say so rather than
# broadcasting one walk across every stratum, which would silently tie them.
function renewal_init_window(
        ::SeedingRandomWalkDraws, I₀::AbstractVector, r, len_gen_int
    )
    return throw(
        ArgumentError(
            "`SeedingRandomWalk` seeds a single series; a stratified renewal " *
                "needs one walk per stratum, which it does not provide"
        )
    )
end

# The specification never reaches a scan through `model()`, which resolves it at
# the pre-scan seam. A hand-built step that skipped that seam lands here.
function renewal_init_window(::SeedingRandomWalk, I₀, r, len_gen_int)
    return error(
        "a `SeedingRandomWalk` still carries its priors, so its walk has not " *
            "been drawn. Resolve the renewal step through `as_turing_model` " *
            "before scanning it."
    )
end

function renewal_init_state(step::SeedingRandomWalkDraws, I₀, r, len_gen_int)
    window = renewal_init_window(step, I₀, r, len_gen_int)
    return (; val = last(window), window = window)
end

# The force-of-infection recursion itself is unchanged — `renewal_foi` is
# generic on `AbstractConstantRenewalStep` and reads `rev_gen_int` and `mixing`
# off the step — so only the window differs from the default core. These two
# methods are needed for the modifier-free case, where `RenewalStep` delegates
# straight to its core.
function (step::SeedingRandomWalkDraws)(state, Rt)
    new_incidence = renewal_foi(step, state.window, Rt)
    return (;
        val = new_incidence,
        window = vcat(state.window[2:end], new_incidence),
    )
end

function get_state(::SeedingRandomWalkDraws, initial_state, state)
    return [s.val for s in state]
end

@doc raw"""
    as_turing_model(step::SeedingRandomWalk, n)

Draw the seeding walk ahead of the scan.

Draws `len_gen_int - 1` standard normals through the `IID` seam and resolves the
step-size slot, returning the [`SeedingRandomWalkDraws`](@ref) the scan seeds
from. A fixed scalar `step_size` is passed straight through, so it costs no
parameter.

# Arguments
- `step`: the [`SeedingRandomWalk`](@ref) core.
- `n`: the length of the renewal series. The walk's own length comes from the
  generation interval, not from this.
"""
@model function as_turing_model(
        step::SeedingRandomWalk{<:Any, <:Any, <:Real}, n
    )
    seed_raw ~ as_turing_submodel(
        IID(Normal()), _n_seed_innovations(step.rev_gen_int); prefix = true
    )
    return SeedingRandomWalkDraws(
        step.rev_gen_int, step.mixing, step.step_size .* seed_raw
    )
end

@model function as_turing_model(step::SeedingRandomWalk, n)
    seed_raw ~ as_turing_submodel(
        IID(Normal()), _n_seed_innovations(step.rev_gen_int); prefix = true
    )
    seed_sd ~ as_turing_submodel(step.step_size, n; prefix = true)
    return SeedingRandomWalkDraws(
        step.rev_gen_int, step.mixing, seed_sd .* seed_raw
    )
end
