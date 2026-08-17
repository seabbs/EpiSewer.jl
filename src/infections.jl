# Infection components: stochastic infections (`infection_noise_estimate`).

@doc raw"""
    InfectionNoise{X} <: AbstractRenewalModifier

Stochastic infections for a renewal process (the `infection_noise_estimate`
component in EpiSewer).

A deterministic renewal process fixes infections at the renewal expectation
``\iota_t``. This modifier gives them their own noise, so infections carry
variance beyond what the reproduction number implies:

```math
\sigma_t^2 = \log\!\left(1 + \frac{1}{\iota_t} + \xi^2\right), \qquad
\log I_t = \log \iota_t - \frac{\sigma_t^2}{2} + \tilde{I}_t \sigma_t,
\qquad \tilde{I}_t \sim \mathrm{Normal}(0, 1)
```

where ``\iota_t = R_t \sum_{i} I_{t-i} g_i`` is the renewal expectation and
``\xi`` is the overdispersion. That makes ``I_t`` log-normal with

```math
\mathbb{E}[I_t] = \iota_t, \qquad
\mathrm{Var}[I_t] = \iota_t \left(1 + \iota_t \xi^2\right),
```

the mean and variance of a negative binomial with mean ``\iota_t`` and
overdispersion ``\xi``. Setting ``\xi = 0`` leaves Poisson variance.

This is the R package's `approx_negative_binomial_log_noncentered`
(`inst/stan/functions/approx_count_dist.stan`), the log-scale moment match it
uses for exactly this purpose, with ``\xi`` fixed at 0.1 by
`infection_noise_estimate()` (`R/model_infections.R`). R's default renewal takes
the same two moments on the linear scale, declaring `I` positive and truncating
its normal at zero
(`inst/stan/EpiSewer_main.stan`, `I ~ normal(iota, ...) T[0, ]`).

Working on the log scale is what makes the component total rather than partial.
Infections are positive by construction, so no truncation is needed and no
proposal can drive them negative: ``I_t = \exp(\cdot) > 0`` always, and
``\iota_t`` is a positive combination of positive infections, so ``\log
\iota_t`` and ``\sigma_t`` are always defined. A linear-scale draw has neither
property, and a negative expected concentration reaching the log-normal
measurement model is an error rather than a rejected proposal.

The parameterisation is **non-centred**: the sampled quantity is the standard
normal ``\tilde{I}_t`` and the location and scale are applied afterwards. That
keeps the prior on the sampled parameter free of ``\iota_t``, which is what makes
the geometry tractable when ``\iota_t`` itself is inferred.

Because the modified incidence is what the renewal scan feeds forward, the noise
compounds through the process rather than perturbing each day in isolation.

# Fields
- `overdispersion`: the prior for ``\xi``, or a fixed scalar. EpiSewer fixes it
  at 0.1 (`overdispersion_prior_sigma = 0`), which is the default here.

# Example
```julia
using EpiSewer, ComposableTuringIDModels, Distributions
r = Renewal(
    [0.2, 0.3, 0.5], EpiSewer.InfectionNoise();
    rt = RandomWalk(), initialisation = Normal()
)
```
"""
struct InfectionNoise{X} <: AbstractRenewalModifier
    "Prior for the overdispersion ``\\xi``, or a fixed scalar."
    overdispersion::X
end

InfectionNoise(; overdispersion = 0.1) = InfectionNoise(overdispersion)

@doc raw"""
    InfectionNoiseDraws{V, X} <: AbstractRenewalModifier

A resolved [`InfectionNoise`](@ref): the drawn standard normals, ready to scan.

This is what [`InfectionNoise`](@ref) returns from its pre-scan
`as_turing_model` seam, following the `ImportedCases` → `ImportedRate` pattern.
Its substate is the step counter, so step ``t`` reads its own
``\tilde{I}_t``.

# Fields
- `raw`: the standard normal draws ``\tilde{I}_t``, one per time.
- `overdispersion`: the resolved ``\xi``.
"""
struct InfectionNoiseDraws{V, X} <: AbstractRenewalModifier
    "The standard normal draws, one per time."
    raw::V
    "The resolved overdispersion."
    overdispersion::X
end

# The substate is the step counter: a scan step has no clock of its own, so a
# per-time modifier carries the index it has reached. The window is unused.
modifier_init_state(::InfectionNoiseDraws, window) = 0

function apply_modifier(mod::InfectionNoiseDraws, incidence, t)
    ξ = mod.overdispersion
    ι = incidence
    # `log1p(1/ι + ξ²)` is R's `log1p_exp(log_sum_exp(-mean_log, 2 log ξ))`,
    # written directly because `ι` is already on the linear scale here.
    σ² = log1p(inv(ι) + ξ^2)
    return exp(log(ι) - σ² / 2 + mod.raw[t + 1] * sqrt(σ²)), t + 1
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
@model function as_turing_model(mod::InfectionNoise{<:Real}, n)
    # A fixed overdispersion is a constant, not a point-mass parameter: that is
    # what `infection_noise_estimate(overdispersion_prior_sigma = 0)` means, and
    # it keeps `ξ` out of the sampled space entirely.
    I_raw ~ as_turing_submodel(IID(Normal()), n; prefix = true)
    return InfectionNoiseDraws(I_raw, mod.overdispersion)
end

@model function as_turing_model(mod::InfectionNoise, n)
    I_raw ~ as_turing_submodel(IID(Normal()), n; prefix = true)
    ξ ~ as_turing_submodel(mod.overdispersion, n; prefix = true)
    return InfectionNoiseDraws(I_raw, ξ)
end
