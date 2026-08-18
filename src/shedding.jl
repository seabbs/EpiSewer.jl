# Shedding components: individual-level load variation (`load_variation_estimate`).

@doc raw"""
    LoadVariation{M, X} <: AbstractObservationModel

Variation in the load shed between individuals (the `load_variation_estimate`
component in EpiSewer).

The expected series ``\lambda_t`` counts shedding individuals. Each sheds a
random load, so the realised load is a **sum** of ``\lambda_t`` individual
draws rather than ``\lambda_t`` times a fixed amount. Summing gamma-distributed
individual loads with coefficient of variation ``\nu`` gives

```math
\zeta_t \sim \mathrm{Gamma}\!\left(\text{mean} = \lambda_t,\;
\mathrm{sd} = \nu \sqrt{\lambda_t}\right),
```

so the load keeps the expected value and gains variance ``\lambda_t \nu^2``.
At ``\nu = 1`` that is Poisson-equivalent variance, which is EpiSewer's default
(`cv_prior_mu = 1` with `cv_prior_sigma = 0` fixes it).

Those are the moments; the draw itself takes them normally and floors the result,

```math
\zeta_t = \mathrm{softplus}\!\left(\lambda_t + \tilde{z}_t \nu
\sqrt{\lambda_t},\; k\right), \qquad \tilde{z}_t \sim \mathrm{Normal}(0, 1)
```

which is R's `gamma_sum_log_approx` exactly
(`inst/stan/functions/dist_gamma_sum.stan`:
`log(softplus(N + noise_noncentered .* sqrt(N) * cv, 10))`). Approximating the
gamma by a normal and flooring it is cheaper than a gamma quantile and, unlike a
gamma moment solve, cannot overflow at a large expectation.

The draw is **non-centred**: the sampled quantity is a standard normal per time
point, so the parameters carry no dependence on ``\lambda_t``.

# Placement
It applies to the shedding-onset series, before the per-case load scaling and
the shedding convolution, which is where Stan applies it
(`omega = log(load_mean) + zeta_log`):

    LatentDelay(LoadVariation(Ascertainment(...)), incubation)

# Fields
- `model`: the wrapped observation model, scoring the varied series.
- `cv`: the individual-level coefficient of variation ``\nu``, or a prior for
  it. EpiSewer fixes it at 1.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = EpiSewer.LoadVariation(NormalError())
```
"""
struct LoadVariation{M <: AbstractObservationModel, X} <: AbstractObservationModel
    "The wrapped observation model."
    model::M
    "Individual-level coefficient of variation."
    cv::X
end

LoadVariation(model::AbstractObservationModel; cv = 1.0) = LoadVariation(model, cv)

"""
    as_turing_model(m::LoadVariation, y_t, Y_t)

Draw the realised load and delegate to the wrapped model.

Samples one standard normal per time point, maps it onto a gamma matched to
mean `Y_t` and standard deviation `cv * sqrt(Y_t)`, and passes the result on.
Returns `(; y_t, expected)` with `expected` the realised load.
"""
@model function as_turing_model(m::LoadVariation, y_t, Y_t)
    load_raw ~ as_turing_submodel(IID(Normal()), length(Y_t); prefix = true)
    # R's `gamma_sum_log_approx`: a normal draw at the gamma's moments, floored
    # by a softplus rather than drawn from a gamma. Cheaper, and it cannot
    # overflow the way a gamma moment solve does at a large expectation.
    floor_k = 10.0
    ζ = _softplus.(
        Y_t .+ load_raw .* sqrt.(max.(Y_t, zero(eltype(Y_t)))) .* m.cv,
        floor_k
    )
    inner ~ as_turing_submodel(m.model, y_t, ζ)
    return (; y_t = inner.y_t, expected = inner.expected)
end
