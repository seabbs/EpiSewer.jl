# Concentration-measurement components: LOD censoring, dPCR noise, LogNormalError.

@doc raw"""
    LOD{E <: AbstractObservationErrorModel, T}

A limit-of-detection (LOD) censored observation model wrapping an underlying
continuous observation-error model (e.g. `NormalError()`).

Writing ``f`` and ``F`` for the density and distribution function of the wrapped
error model at expected value ``Y_t``, and ``L`` for the detection limit, an
observation contributes

```math
\log p(y_t) = \begin{cases}
  \log F(L)   & y_t = L \\
  \log f(y_t) & y_t > L \\
  -\infty     & y_t < L
\end{cases}
```

A measurement at the limit therefore scores the probability of being anywhere at
or below it, rather than a density. This is the data convention EpiSewer uses:
non-detects are reported *at* the limit, as in its example data, so `y_t == L`
means "somewhere below ``L``" rather than "exactly ``L``". `missing` entries are
handled by the standard `AbstractObservationErrorModel` loop.

# Fields
- `error_model`: the underlying observation-error model.
- `lod::T`: the detection limit.

# Example
```julia
using EpiSewer, Distributions
m = EpiSewer.LOD(NormalError(); lod = 50.0)
model = as_turing_model(m, [50.0, 120.0, missing], fill(100.0, 3))
```
"""
struct LOD{E <: AbstractObservationErrorModel, T} <: AbstractObservationErrorModel
    error_model::E
    lod::T
end

LOD(error_model::AbstractObservationErrorModel; lod::Real) = LOD(error_model, lod)

# Convenience constructor matching EpiSewer's noise-observation default.
LOD(; lod::Real = 0.0, std = HalfNormal(0.1)) = LOD(NormalError(; std = std), lod)

generate_observation_error_priors(m::LOD, y_t, Y_t) =
    generate_observation_error_priors(m.error_model, y_t, Y_t)

# Left-censor the inner per-time-point error distribution at `lod`.
function observation_error(m::LOD, Y_t, priors...)
    return censored(observation_error(m.error_model, Y_t, priors...), m.lod, Inf)
end

@doc raw"""
    DigitalPCRError{T <: AbstractVector{<:Integer}}

A digital PCR observation model: the positive partition counts are scored
directly, rather than the concentration derived from them.

With ``Y_t`` the log expected copies per partition, the copies landing in one
partition are Poisson with mean ``e^{Y_t}``, so a partition tests positive
whenever it receives at least one copy:

```math
p_t = 1 - \exp(-e^{Y_t}), \qquad
y_t \sim \mathrm{Binomial}(N_t, p_t)
```

The link ``Y_t \mapsto 1 - \exp(-e^{Y_t})`` is the inverse of the
complementary log-log link, and it is the right one here because it *is* the
Poisson zero-probability: ``1 - e^{-\lambda}`` is the chance of a non-empty
partition at mean ``\lambda``. Saturation follows for free, since ``p_t \to 1``
as the expected copies grow.

The component is a composition of ecosystem pieces:
`TransformObservationModel`
applies the cloglog-inverse link to the expected series and
`BinomialError` scores the counts.
The data contract follows `BinomialError`: `y_t = (y = positives, N = totals)`.
`N` is supplied by the component itself, so a bare vector of positive counts is
accepted too and the totals in a `NamedTuple` are redundant.

A `missing` count is imputed: it is drawn as a `y_t` parameter rather than
scored, and the observed counts alone form the likelihood.

# Fields
- `total_partitions::T`: valid partition count per measurement.

# Example
```julia
dpcr = EpiSewer.DigitalPCRError([1000, 1000, 1000])
Y = log.([0.01, 0.02, 0.03])  # log expected copies per partition
mdl = as_turing_model(dpcr, [10, 25, missing], Y)
rand(mdl)                     # the missing day comes back as a drawn `y_t[3]`
```
"""
struct DigitalPCRError{T <: AbstractVector{<:Integer}} <: AbstractObservationModel
    total_partitions::T
end

_transformed_dpcr(m::DigitalPCRError) = TransformObservationModel(
    BinomialError(),
    x -> 1.0 .- exp.(-exp.(x)),
)

@model function _as_turing_model_dpcr(m::DigitalPCRError, y_t, Y_t)
    inner ~ as_turing_submodel(_transformed_dpcr(m), y_t, Y_t)
    return (; y_t = inner.y_t, expected = inner.expected)
end

# The partition totals are folded into the data here, and the result is narrowed
# by `concrete_observations` before it reaches the submodel.
#
# The narrowing is what makes a `missing` count safe. `BinomialError`'s data is
# a `NamedTuple`, and an array inside a `NamedTuple` argument is not something
# DynamicPPL sees as a model argument: the scoring loop's `~` then writes the
# imputed draw straight into the caller's vector and registers no `y_t`
# parameter at all, so every evaluation after the first scores that draw as data
# (EpiAware/ComposableTuringIDModels.jl#264). Narrowing splits the vector into a
# `MissingObservations` carrier instead, whose scoring path copies the values and
# assumes the absent ones. Because this component always builds the `NamedTuple`
# itself, the hazard applies whatever the caller passes, so the guard belongs
# here rather than on the caller.
#
# It sits outside the `@model` body, as `as_turing_model(::IDModel, ...)` does,
# so it runs once per model rather than once per log-density evaluation and
# stays off the differentiated path. Re-narrowing already-narrowed data returns
# the same objects, so the `IDModel` path applying it first costs nothing.
#
# Called qualified because `concrete_observations` is documented upstream but not
# declared public, and the quality suite rejects an explicit import of a
# non-public name.
function as_turing_model(m::DigitalPCRError, y_t, Y_t)
    y = y_t isa NamedTuple ? merge(y_t, (N = m.total_partitions,)) :
        (y = y_t, N = m.total_partitions)
    return _as_turing_model_dpcr(
        m, ComposableTuringIDModels.concrete_observations(y), Y_t
    )
end

@doc raw"""
    LogNormalError{S <: PriorLike}

A log-normal observation-error model with an inferred coefficient of variation.

The measurement noise is relative: the distribution's real-space mean is the
expected concentration and its real-space standard deviation is proportional to
it, so ``\sigma`` is a coefficient of variation rather than an absolute scale.

```math
\mathbb{E}[y_t] = Y_t, \qquad \mathrm{sd}[y_t] = \sigma Y_t, \qquad
y_t \sim \mathrm{LogNormal}(\mu_t, s)
```

A `LogNormal` is parameterised on the log scale, so those two moments are
converted to its native parameters:

```math
s^2 = \log\!\left(1 + \sigma^2\right), \qquad
\mu_t = \log Y_t - s^2 / 2
```

`reparameterise` from ReparameterisedDistributions performs that conversion, so
the component states the moments it means and the log-scale parameters follow.
Note ``s`` does not depend on ``Y_t``: a constant coefficient of variation is a
constant variance on the log scale, which is what makes this the relative-noise
family. This matches EpiSewer's `noise_estimate` convention.

`cv` sets the prior for ``\sigma`` — a constant `Distribution` or a length-`n`
process, drawn through the `as_turing_submodel` seam like `NormalError`
treats its `std` prior. It is sampled under the name `cv`, which keeps it clear
of the bare `σ` that the Gaussian-process latent models use for their marginal
standard deviation.

# Fields
- `cv::S`: prior for the coefficient of variation `σ`.

# Example
```julia
using EpiSewer, Distributions
lne = EpiSewer.LogNormalError()
mdl = as_turing_model(lne, missing, fill(100.0, 10))
rand(mdl)
```
"""
struct LogNormalError{S <: PriorLike} <: AbstractObservationErrorModel
    "Prior for the coefficient of variation."
    cv::S
end

LogNormalError(; cv = HalfNormal(0.1)) = LogNormalError(cv)

@model function generate_observation_error_priors(
        obs_model::LogNormalError, y_t, Y_t
    )
    # Sampled as `cv`, not `σ`: the Gaussian-process latent models sample a bare
    # `σ` for their marginal standard deviation, and a second `σ` in the same
    # chain silently overwrites the first (upstream ComposableTuringIDModels
    # issue #268). `cv` is also the more accurate name for a coefficient of
    # variation.
    cv ~ as_turing_submodel(obs_model.cv, length(Y_t); prefix = true)
    return (; cv = cv)
end

function observation_error(::LogNormalError, Y_t, σ)
    # Three guards covering disjoint failures, all reachable from a diverging
    # sampler.
    #
    # A non-finite `Y_t` or `σ` returns a plain `LogNormal` sentinel, whose
    # `logcdf` is `-Inf` as well as its `logpdf`. That is what keeps the result
    # safe to left-censor in `LOD`: a censored distribution's boundary term is a
    # `logcdf`, and `Reparameterised` guards only `logpdf`/`pdf`, routing
    # `logcdf` through `native`, which throws (see the boundary test).
    # A bare `Float64` `-Inf` is deliberate here — giving the sentinel the
    # input's type instead makes AD return a `NaN` partial rather than `0.0`.
    isfinite(Y_t) && isfinite(σ) || return LogNormal(Inf, 1.0)
    # An enormous but finite coefficient of variation. The log-scale conversion
    # is `s² = log1p((sd / mean)²)`, and `sd / mean` is exactly `σ` here, so
    # `σ²` overflows to `Inf` once `σ` passes `sqrt(floatmax)` ≈ 1.3e154 — and
    # `Inf` reaches the constructor as a valid-looking argument rather than as a
    # non-finite one, so the guard above does not catch it. Sampling a `missing`
    # observation calls `rand`, which validates through `native` and throws,
    # so an extreme proposal has to be turned into `-Inf` here rather than
    # rejected downstream.
    σ < sqrt(floatmax(Float64)) || return LogNormal(Inf, 1.0)
    # Finite but invalid moments (`Y_t <= 0`, `σ <= 0`) reach `Reparameterised`,
    # whose `logpdf` scores them `-Inf` at the input's own type. `check_args =
    # false` is what routes us there: the checking constructor validates the
    # moments and would throw a `DomainError` mid-gradient instead.
    return reparameterise(LogNormal; mean = Y_t, sd = σ * Y_t, check_args = false)
end
