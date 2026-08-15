# Discretised distribution helpers for the model assumptions.
#
# These reproduce EpiSewer's `get_discrete_gamma()` and
# `get_discrete_gamma_shifted()` helpers using CensoredDistributions.jl
# primitives, so the daily PMFs consumed by `Renewal` (generation interval)
# and `LatentDelay` (incubation / shedding load) are built on demand from
# continuous distributions rather than stored on disk.

using CensoredDistributions
using Distributions

# Internal: the non-shifted discrete Gamma PMF, matching EpiSewer's
# `get_discrete_gamma()` / `extraDistr::ddgamma` convention.
#
# EpiSewer's discretisation places probability mass in unit-width bins
# [k, k+1) for k = 0, 1, ..., maxX-1, with the remainder ("longest" bin,
# [maxX, ∞)) appended at the end, i.e. `include_zero = TRUE`:
#
#   probs = [ddgamma(0:(maxX-1)), 1 - pgamma(maxX)]
#
# The statistically principled daily discretisation is
# `double_interval_censored(dist; interval = 1)`: the event time is both
# PRIMARY-censored (the exposure/primary event is averaged over its day via
# a `Uniform(0, 1)`) AND SECONDARY interval-censored (the observation falls
# in a unit day bin [k, k+1)). Plain `interval_censored` (only the daily
# bin) under-accounts for the within-day timing of the primary event;
# `double_interval_censored` is the standard doubly-discretised daily delay
# PMF (Park 2024; Charniga 2024; equivalently the EpiEstim/Cori et al.
# closed form for a shifted Gamma). `pdf(d, k)` is the mass in [k, k+1),
# and the final bin pools the tail [maxX, ∞) as `1 - cdf(d, maxX)`.
function _discrete_gamma(shape::Real, scale::Real; maxX::Int = 40)
    d = double_interval_censored(
        Gamma(shape, scale), interval = 1
    )
    probs = [pdf(d, k) for k in 0:(maxX - 1)]
    tail = 1.0 - cdf(d, maxX)
    return vcat(probs, tail)
end

"""
    get_discrete_gamma(; shape = nothing, scale = nothing, mean = nothing,
        sd = nothing, maxX = 40)

Discretised Gamma PMF (daily bins), reproducing EpiSewer's
`get_discrete_gamma()`.

The Gamma is parameterised either directly by `shape`/`scale`, or by
`mean`/`sd` (with `shape = (mean / sd)^2`, `scale = sd^2 / mean`, as in
EpiSewer's `get_gamma_shape_alternative` / `get_gamma_scale_alternative`).

The daily bin probabilities are computed with
[`double_interval_censored`](@ref CensoredDistributions.double_interval_censored):
the delay is PRIMARY-censored (primary event averaged over its day) and
SECONDARY interval-censored into daily bins `[k, k+1)` for `k = 0:(maxX-1)`,
with the tail `[maxX, ∞)` pooled into the final bin. This is the
statistically correct doubly-discretised daily-delay PMF, rather than plain
unit-bin CDF differences.

# Arguments
- `shape`, `scale`: Gamma shape and scale. Alternatively `mean` and `sd`.
- `maxX::Int = 40`: index of the final tail bin; all mass at/above `maxX`
  is pooled into it.

# Example
```julia
pmf = get_discrete_gamma(mean = 8.5, sd = 0.4)  # incubation period
```
"""
function get_discrete_gamma(;
        shape::Union{Real, Nothing} = nothing,
        scale::Union{Real, Nothing} = nothing,
        mean::Union{Real, Nothing} = nothing,
        sd::Union{Real, Nothing} = nothing,
        maxX::Int = 40,
    )
    if shape === nothing && scale === nothing
        @assert mean !== nothing && sd !== nothing "provide shape/scale or mean/sd"
        shape = (mean / sd)^2
        scale = sd^2 / mean
    end
    @assert shape !== nothing && scale !== nothing "provide shape/scale or mean/sd"
    return _discrete_gamma(shape, scale; maxX = maxX)
end

"""
    get_discrete_gamma_shifted(mean, sd; maxX = 40)

Discretised PMF of a shifted Gamma (minimum at 1), reproducing EpiSewer's
`get_discrete_gamma_shifted()`.

The shifted Gamma is the standard choice for a generation-time distribution
(zero probability of a zero generation time, as required by the renewal
equation). The parameters are reparameterised as

```math
a = \\left(\\frac{\\mu - 1}{\\sigma}\\right)^2, \\qquad
b = \\frac{\\sigma^2}{\\mu - 1}
```

and the daily bin probabilities are computed with the doubly-discretised
shifted-Gamma formula used by EpiSewer (itself adapted from
EpiEstim/Cori et al.), expressed here through
[`double_interval_censored`](@ref CensoredDistributions.double_interval_censored):
`P(T = k)` is the mass of `double_interval_censored(Gamma(a, b); interval = 1)`
in the day bin `[k-1, k)`, normalised to sum to one. This is the same
mathematics as the closed-form EpiEstim formula — primary within-day
averaging plus daily interval censoring — implemented with the ecosystem's
CensoredDistributions primitives.

# Arguments
- `mean`, `sd`: mean and standard deviation of the shifted Gamma.
- `maxX::Int = 40`: right truncation index for the PMF.

# Example
```julia
pmf = get_discrete_gamma_shifted(3.0, 2.4)  # generation time
```
"""
function get_discrete_gamma_shifted(mean::Real, sd::Real; maxX::Int = 40)
    @assert mean > 1 "gamma_mean must be > 1"
    @assert sd >= 0 "gamma_sd must be >= 0"
    @assert maxX >= 1 "maxX must be >= 1"

    a = ((mean - 1) / sd)^2
    b = sd^2 / (mean - 1)
    k = collect(1.0:maxX)

    # Day k (k >= 1) holds the mass of the doubly-discretised Gamma in
    # [k-1, k): `pdf(d, k-1)` for d = double_interval_censored(Gamma(a, b);
    # interval = 1). This is the primary- plus daily-interval-censored delay
    # PMF, the ecosystem expression of the EpiEstim closed form below.
    d = double_interval_censored(Gamma(a, b), interval = 1)
    res = [pdf(d, kk - 1) for kk in k]
    res = max.(res, 0)
    return res ./ sum(res)
end
