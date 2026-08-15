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
# `interval_censored(Gamma(shape, scale), 1)` computes `pdf(k) =
# cdf(k + 1) - cdf(k)` per unit bin, which is exactly `ddgamma(k)`.
# The final tail bin is `1 - cdf(maxX)`.
function _discrete_gamma(shape::Real, scale::Real; maxX::Int = 40)
    d = interval_censored(Gamma(shape, scale), 1)
    probs = [pdf(d, k) for k in 0:(maxX - 1)]
    tail = 1.0 - cdf(Gamma(shape, scale), maxX)
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
The PMF is the unit-interval probability mass `[cdf(k+1) - cdf(k)]` for
`k = 0:(maxX-1)` plus the tail bin `[maxX, ∞)`, matching the R implementation
(`extraDistr::ddgamma` with `include_zero = TRUE`).

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

and the daily bin probabilities are computed with the closed-form
doubly-discretised shifted-Gamma CDF combination used by EpiSewer (itself
adapted from EpiEstim/Cori et al.), normalised to sum to one.

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

    # Closed-form bin probabilities for the doubly-discretised shifted Gamma
    # (EpiSewer / EpiEstim formula):
    #   P(T = k) = k C(k) + (k-2) C(k-2) - 2(k-1) C(k-1)
    #              + a b (2 C(k-1; a+1) - C(k-2; a+1) - C(k; a+1))
    # where C(x) = cdf(Gamma(a, b), x) and C(x; a+1) = cdf(Gamma(a+1, b), x).
    C(x) = cdf(Gamma(a, b), x)
    Cp(x) = cdf(Gamma(a + 1, b), x)  # shape a+1, same scale b

    res = k .* C.(k) .+ (k .- 2) .* C.(k .- 2) .- 2 .* (k .- 1) .* C.(k .- 1)
    res = res .+ a .* b .* (2 .* Cp.(k .- 1) .- Cp.(k .- 2) .- Cp.(k))
    res = max.(res, 0)
    return res ./ sum(res)
end
