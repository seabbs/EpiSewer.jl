# Infection components: stochastic infections (`infection_noise_estimate`).

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
