# Infection components: stochastic infections (`infection_noise_estimate`).

# The moment-matched noise distribution. `to_native` is
# ReparameterisedDistributions' own moment solve, so any family it registers for
# `(:mean, :sd)` works here — `LogNormal`, `Gamma`, `InverseGaussian` and the
# rest — and the solve is theirs rather than a re-derivation.
#
# `Normal` is native in `(:mean, :sd)` already, which is why the package does not
# register it. Dispatching here rather than adding a `to_native` method keeps the
# special case ours: a method on their function for their type would be piracy.
_moment_match(::Type{Normal}, mean, sd) = Normal(mean, sd)
_moment_match(family, mean, sd) = to_native(family, Val{(:mean, :sd)}(), (mean, sd))

# The whole draw in one step, for the families whose moment solve and quantile
# are both closed form. This exists for speed rather than for correctness: the
# generic path constructs (and validates) a distribution object per time step
# inside the differentiated scan, which is 164 allocations per gradient on the
# example series and roughly doubles its cost. `_draw` and the generic fallback
# agree to machine precision, which `test/infections.jl` asserts.
function _draw(::Type{LogNormal}, mean, sd, z)
    σ² = log1p((sd / mean)^2)
    return exp(log(mean) - σ² / 2 + z * sqrt(σ²))
end
_draw(::Type{Normal}, mean, sd, z) = mean + sd * z
_draw(family, mean, sd, z) = _noncentred(_moment_match(family, mean, sd), z)

# The non-centred draw: map a standard normal onto the matched distribution.
#
# Both location-scale cases are written out rather than routed through the
# generic fallback, because the fallback round-trips through `cdf` and
# `quantile` and loses the tails: a draw five standard deviations out underflows
# `cdf` to 0 and comes back as the support's endpoint. A sampler does reach five
# standard deviations.
_noncentred(d::Normal, z) = d.μ + d.σ * z
_noncentred(d::LogNormal, z) = exp(d.μ + d.σ * z)
_noncentred(d, z) = quantile(d, cdf(Normal(), z))

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
is `ReparameterisedDistributions.to_native`, so any family registered for
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

The default `LogNormal` keeps infections **positive by construction**, so no
truncation is needed and no proposal can drive the expected concentration
negative: ``I_t = \exp(\cdot) > 0`` always, and ``\iota_t`` is a positive
combination of positive infections, so ``\log \iota_t`` and ``\sigma_t`` are
always defined. `Normal` is R's own choice and reproduces it exactly, but is only
non-negative because R declares `vector<lower=0> I` and truncates
(`inst/stan/EpiSewer_main.stan`: `I ~ normal(iota, ...) T[0, ]`); reached through
this modifier it has no such bound, and a negative expected concentration is an
error rather than a rejected proposal in a log-normal measurement model.

``c_t`` is the coefficient of variation ``\sqrt{1/\iota_t + \xi^2}`` under a
smooth upper limit, ``u - \mathrm{softplus}(u - c, k)`` with ``u`` = `cv_cap` and
``k`` = `cv_sharpness`. The limit tracks ``c`` closely well below ``u`` and
saturates above it, so the moments hold wherever it is inactive. Without it the
coefficient of variation diverges as ``\iota_t`` approaches zero, giving an
arbitrarily small expectation an arbitrarily wide draw. R applies the same limit
with the same constants
(`soft_upper(sqrt(iota .* (1 + iota * I_xi^2)) ./ iota, 0.5, 10)`), which is why
its own comment calls the result an approximation to a negative binomial. Setting
`cv_cap = Inf` restores the exact negative-binomial variance.

The parameterisation is **non-centred**: the sampled quantity is the standard
normal ``\tilde{I}_t`` and the location and scale are applied afterwards. This is
forced rather than chosen — `apply_modifier` is deterministic and a modifier's
priors resolve before the scan, so a modifier cannot draw ``I_t`` conditional on
``\iota_t`` (ComposableTuringIDModels issue #271).

Because the modified incidence is what the renewal scan feeds forward, the noise
compounds through the process rather than perturbing each day in isolation.

# Effect on sampling
The noise earns its parameters. Measured on the thinned example series, 100
warmup + 100 draws under `NUTS(0.9)` with Mooncake, against the same model with
`infection_noise = nothing`:

| infection noise | mean tree depth | divergences |
|:---|---:|---:|
| none | 3.50 | 85 |
| this component | 10.00 | 0 |

A deterministic renewal makes every observation an exact function of ``R_t`` and
the seeding, and that stiffness is what the divergences are. The cost is
trajectory length: maximum tree depth, so 1024 leapfrog steps an iteration. R
reaches zero divergences *and* zero maximum-treedepth hits, and the difference
there is the centred parameterisation its Stan code can express and a modifier
cannot.

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
end

# The substate is the step counter: a scan step has no clock of its own, so a
# per-time modifier carries the index it has reached. The window is unused.
modifier_init_state(::InfectionNoiseDraws, window) = 0

# R's `softplus(x, k) = log1p_exp(k x) / k` and
# `soft_upper(x, u, k) = u - softplus(u - x, k)`
# (`.resources/EpiSewer/inst/stan/functions/link.stan`): a smooth limit that
# tracks `x` well below `u` and saturates at `u` above it.
#
# An infinite limit is no limit: taken literally it would give `Inf - Inf`, so
# the uncapped case returns `x`. `u` and `k` are construction-time constants, so
# this branch is never on a differentiated value.
# Numerically stable: the naive `log1p(exp(k x)) / k` overflows to `Inf` once
# `k x` passes about 709, and both the `R_t` link and the load-variation floor
# see values well beyond that on a diverging proposal. This form is exact and
# saturates to `x` instead.
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
