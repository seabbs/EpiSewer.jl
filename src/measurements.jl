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

# Narrow the data before it reaches the submodel. An array inside a `NamedTuple`
# argument is not a model argument to DynamicPPL, so without this the scoring
# loop writes an imputed draw into the caller's vector and scores it as data on
# every later evaluation (ComposableTuringIDModels#264). Narrowing splits it into
# a `MissingObservations` carrier instead. It sits outside the `@model` so it
# runs once per model rather than once per gradient.
function as_turing_model(m::DigitalPCRError, y_t, Y_t)
    y = y_t isa NamedTuple ? merge(y_t, (N = m.total_partitions,)) :
        (y = y_t, N = m.total_partitions)
    return _as_turing_model_dpcr(
        m, ComposableTuringIDModels.concrete_observations(y), Y_t
    )
end

@doc raw"""
    GammaError{S <: PriorLike}

A gamma observation-error model with an inferred coefficient of variation.

This is EpiSewer's default concentration likelihood: `noise_estimate`'s
`distribution` argument defaults to `"gamma"`, of the four families it offers.
Like [`LogNormalError`](@ref EpiSewer.LogNormalError) the noise is relative, so
``\sigma`` is a coefficient of variation rather than an absolute scale:

```math
\mathbb{E}[y_t] = Y_t, \qquad \mathrm{sd}[y_t] = \sigma Y_t, \qquad
y_t \sim \mathrm{Gamma}(\alpha, \beta)
```

`reparameterise` solves those moments for the native shape and rate, which
recovers R's `gamma3_lpdf` exactly:

```math
\alpha = \frac{1}{\sigma^2}, \qquad \beta = \frac{1}{Y_t \sigma^2}
```

The two families differ in their tails at the same moments. A gamma is the
lighter-tailed of the pair, so it is less forgiving of a measurement far above
the expected concentration — which is what R's outlier component exists to
absorb.

# Fields
- `cv::S`: prior for the coefficient of variation `σ`. Defaults to R's, as
  [`LogNormalError`](@ref EpiSewer.LogNormalError) does.

# Example
```julia
using EpiSewer, Distributions
ge = EpiSewer.GammaError()
mdl = as_turing_model(ge, missing, fill(100.0, 10))
rand(mdl)
```
"""
struct GammaError{S <: PriorLike} <: AbstractObservationErrorModel
    "Prior for the coefficient of variation."
    cv::S
end

GammaError(; cv = HalfNormal(sqrt(2 / π))) = GammaError(cv)

@model function generate_observation_error_priors(
        obs_model::GammaError, y_t, Y_t
    )
    cv ~ as_turing_submodel(obs_model.cv, length(Y_t); prefix = true)
    return (; cv = cv)
end

function observation_error(::GammaError, Y_t, σ)
    # The sentinel has to satisfy two things at once, and only one candidate
    # does. `logpdf` must be `-Inf` so a diverging proposal is rejected, and
    # `rand` must not throw, because imputing a `missing` observation samples
    # from whatever this returns. `Gamma(Inf, 1)` scores `NaN`, which poisons
    # the gradient. An invalid moment pair under `check_args = false` scores
    # `-Inf` correctly but throws a `DomainError` from `rand`, which only
    # surfaces when a fit reaches an unusable point with data missing.
    # `Gamma(1, Inf)` does both: `-Inf` and a finite-free `Inf` draw. It is the
    # counterpart of `LogNormal(Inf, 1)` in `LogNormalError`.
    _reject = Gamma(1.0, Inf)
    (isfinite(Y_t) && isfinite(σ)) || return _reject
    (Y_t > 0 && σ > 0) || return _reject
    return reparameterise(Gamma; mean = Y_t, sd = σ * Y_t, check_args = false)
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
treats its `std` prior. It is sampled under the name `cv`, which is what a
coefficient of variation is.

The default matches R's `noise_estimate(cv_prior_mu = 0, cv_prior_sigma = 1)`,
which Stan scores as `normal(0, 1) T[0, ]` — a half-normal of scale 1, so a mean
of ``\sqrt{2/\pi}``. `HalfNormal` here is parameterised by its **mean**, not
its scale, hence `HalfNormal(sqrt(2 / π))` rather than `HalfNormal(1)`.

# Fields
- `cv::S`: prior for the coefficient of variation `σ`. Defaults to R's.

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

# R's own prior, not its initial value. `noise_estimate`'s default is
# `cv_prior_sigma = 1` (`R/model_measurements.R:547`), scored at
# `EpiSewer_main.stan:910` as `normal(0, 1) T[0, ]`. The 0.1 this replaced is
# R's *init* for the same parameter (`model_measurements.R:591`, commented "10%
# coefficient of variation"), which is a different thing and eight times
# tighter than the prior. A tight noise prior leaves the outlier spikes to
# explain any deviation the noise cannot absorb.
LogNormalError(; cv = HalfNormal(sqrt(2 / π))) = LogNormalError(cv)

@model function generate_observation_error_priors(
        obs_model::LogNormalError, y_t, Y_t
    )
    # Sampled as `cv`, which is what a coefficient of variation is.
    cv ~ as_turing_submodel(obs_model.cv, length(Y_t); prefix = true)
    return (; cv = cv)
end

function observation_error(::LogNormalError, Y_t, σ)
    # Three guards, all reachable from a diverging sampler. The sentinel's
    # `logcdf` is `-Inf` as well as its `logpdf`, which keeps it safe to
    # left-censor in `LOD`. A bare `Float64` `-Inf` is deliberate: giving the
    # sentinel the input's type makes AD return `NaN` rather than `0.0`.
    isfinite(Y_t) && isfinite(σ) || return LogNormal(Inf, 1.0)
    # `s² = log1p(σ²)` overflows once `σ` passes `sqrt(floatmax)`, and the
    # resulting `Inf` looks like a valid argument to the guard above. Sampling a
    # `missing` observation calls `rand`, which would then throw.
    σ < sqrt(floatmax(Float64)) || return LogNormal(Inf, 1.0)
    # `check_args = false` routes invalid moments to a `-Inf` score rather than
    # a `DomainError` mid-gradient.
    return reparameterise(LogNormal; mean = Y_t, sd = σ * Y_t, check_args = false)
end
