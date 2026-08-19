using EpiSewer
using TestItemRunner

@testitem "LOD constructs and wraps a NormalError" begin
    using EpiSewer

    m = EpiSewer.LOD(
        EpiSewer.ComposableTuringIDModels.NormalError(); lod = 50.0
    )
    @test m.lod == 50.0
    @test m.error_model isa EpiSewer.ComposableTuringIDModels.NormalError
end

@testitem "LOD default convenience constructor" begin
    using EpiSewer

    m = EpiSewer.LOD()
    @test m.lod == 0.0
end

@testitem "LOD censored loglik matches logcdf of the underlying error distribution" begin
    using EpiSewer
    using Distributions: Distributions, Normal, logcdf
    using ForwardDiff

    # The two claims about a censored measurement — that it is reported AT the
    # limit, and that it then scores `logcdf` rather than a density — are pinned
    # here at the distribution and in `integration_tests.jl` at the model, by
    # "LOD composed in an IDModel left-censors at the limit".
    CT = EpiSewer.ComposableTuringIDModels
    # A fixed observation-noise sigma (degenerate Normal prior) so the censored
    # likelihood is exactly logcdf(Normal(100, 1), 50).
    m = EpiSewer.LOD(CT.NormalError(; std = Normal(1.0, 0.0)); lod = 50.0)
    # σ = 1.0 is passed as the observation-error prior; the censored dist at
    # the boundary is Censored(Normal(100, 1), 50, Inf).
    obs_err = CT.observation_error(m, 100.0, 1.0)
    # Data at the LOD scores logcdf(Normal(100, 1), 50).
    @test isapprox(
        Distributions.logpdf(obs_err, 50.0),
        logcdf(Normal(100.0, 1.0), 50.0);
        atol = 1.0e-9,
    )
    # The censored contribution is AD-differentiable w.r.t. the expected value.
    f(mu) = Distributions.logpdf(
        CT.observation_error(
            EpiSewer.LOD(CT.NormalError(; std = Normal(1.0, 0.0)); lod = 50.0), mu, 1.0
        ), 50.0
    )
    @test isfinite(ForwardDiff.derivative(f, 100.0))
end

@testitem "LogNormalError parameterises the CV of the measurement error" begin
    using EpiSewer
    using Distributions: Distributions, mean, std, insupport
    using ForwardDiff

    # σ is the coefficient of variation: the real-space mean is Y_t and the
    # real-space sd is σ · Y_t, whatever the scale of Y_t. `observation_error`
    # returns a `ReparameterisedDistributions.Reparameterised` over `LogNormal`,
    # so the contract is tested through the distribution interface rather than
    # the concrete type.
    for Y_t in (1.0, 100.0, 3.0e3), σ in (0.05, 0.3)
        d = EpiSewer.observation_error(EpiSewer.LogNormalError(), Y_t, σ)
        @test d isa Distributions.Distribution
        @test mean(d) ≈ Y_t rtol = 1.0e-8
        @test std(d) / mean(d) ≈ σ rtol = 1.0e-8
        # Log-normal support: positive concentrations only.
        @test insupport(d, Y_t)
        @test !insupport(d, -1.0)
    end

    # The log-density is differentiable w.r.t. the expected concentration.
    f(Y_t) = Distributions.logpdf(
        EpiSewer.observation_error(EpiSewer.LogNormalError(), Y_t, 0.1), 95.0
    )
    @test isfinite(ForwardDiff.derivative(f, 100.0))
end

@testitem "LogNormalError rejects a non-finite expected value instead of throwing" begin
    using EpiSewer
    using Distributions: Distributions

    # An exploding latent draw must score -Inf, not raise out of the moment
    # validation in `reparameterise`.
    for bad in (Inf, NaN)
        d = EpiSewer.observation_error(EpiSewer.LogNormalError(), bad, 0.1)
        @test Distributions.logpdf(d, 100.0) == -Inf
    end
    d = EpiSewer.observation_error(EpiSewer.LogNormalError(), 100.0, Inf)
    @test Distributions.logpdf(d, 100.0) == -Inf
end

@testitem "LogNormalError scores finite-but-invalid moments as -Inf" begin
    using EpiSewer
    using Distributions: Distributions

    # A non-positive expected concentration or CV is finite, so it passes the
    # non-finite guard and is rejected by `Reparameterised`'s own moment
    # validation rather than throwing out of the constructor.
    for (Y_t, σ) in ((-1.0, 0.1), (0.0, 0.1), (100.0, -0.1), (100.0, 0.0))
        d = EpiSewer.observation_error(EpiSewer.LogNormalError(), Y_t, σ)
        @test Distributions.logpdf(d, 100.0) == -Inf
    end
end

@testitem "LogNormalError rejects finite-invalid moments at the input's type" begin
    using EpiSewer
    using Distributions: Distributions
    using ForwardDiff

    # On the `check_args = false` path the rejection is `Reparameterised`'s own,
    # so `-Inf` comes back as a `Dual` rather than a bare `Float64` that would
    # break a gradient tape.
    d = EpiSewer.observation_error(
        EpiSewer.LogNormalError(), ForwardDiff.Dual(-1.0, 1.0), 0.1
    )
    lp = Distributions.logpdf(d, 100.0)
    @test lp isa ForwardDiff.Dual
    @test ForwardDiff.value(lp) == -Inf
end

@testitem "LOD(LogNormalError()) scores -Inf at the LOD on a non-finite draw" begin
    using EpiSewer
    using Distributions: Distributions

    # Regression test. `Censored`'s boundary term is a `logcdf`, and
    # `Reparameterised` guards only `logpdf`/`pdf` — its `logcdf` routes through
    # `native`, which throws. The non-finite sentinel in `observation_error` is
    # a plain `LogNormal`, so both terms score `-Inf` and the composition stays
    # safe to censor. `y_t == lod` is the documented data convention, so the
    # boundary is the reachable case, not a corner.
    for bad in (Inf, NaN)
        d = EpiSewer.observation_error(
            EpiSewer.LOD(EpiSewer.LogNormalError(); lod = 50.0), bad, 0.1
        )
        @test Distributions.logpdf(d, 50.0) == -Inf
        @test Distributions.logpdf(d, 100.0) == -Inf
    end

    # A finite-but-invalid draw still reaches the unguarded `logcdf`, so the
    # boundary raises there. Left broken deliberately: the fix belongs upstream,
    # not in a third guard here. Tracked as
    # EpiAware/ReparameterisedDistributions.jl#115, which asks for
    # `cdf`/`logcdf`/`ccdf`/`logccdf` to be guarded by `valid_moments` the way
    # `logpdf`/`pdf` already are. Retire this `@test_broken` when that lands.
    d = EpiSewer.observation_error(
        EpiSewer.LOD(EpiSewer.LogNormalError(); lod = 50.0), -1.0, 0.1
    )
    @test Distributions.logpdf(d, 100.0) == -Inf
    @test_broken Distributions.logpdf(d, 50.0) == -Inf
end

@testitem "DigitalPCRError imputes a missing partition count" begin
    using EpiSewer
    using Turing, Distributions, Random
    import ComposableTuringIDModels as CT

    m = EpiSewer.DigitalPCRError([1000, 1000, 1000])
    @test m.total_partitions == [1000, 1000, 1000]

    # Data contract: NamedTuple (y = positive partitions, N = total partitions).
    positives = Union{Missing, Int}[10, 25, missing]
    Y = log.([0.01, 0.02, 0.03])
    mdl = CT.as_turing_model(m, (y = positives, N = m.total_partitions), Y)

    # The `missing` day is a parameter, not data: it is imputed as a partition
    # count and takes a different value in each draw.
    #
    # This is what the component's `concrete_observations` guard buys. Without
    # it the count vector reaches the scoring loop inside a `NamedTuple`, which
    # DynamicPPL does not see as a model argument, so the loop's `~` wrote the
    # draw into the caller's own vector and registered no parameter at all: the
    # chain carried no `y_t`, and every evaluation after the first scored that
    # one draw as data (EpiAware/ComposableTuringIDModels.jl#264).
    Random.seed!(11)
    chn = sample(mdl, Prior(), 40; progress = false)
    imputed = vec(chn[@varname(y_t[3])])
    @test length(imputed) == 40
    @test length(unique(imputed)) > 1
    @test all(0 .<= imputed .<= 1000)
    # The observed days are untouched, and so is the caller's vector.
    @test ismissing(positives[3])
    @test positives[1:2] == [10, 25]

    # A bare count vector is the same model: the component supplies `N`.
    Random.seed!(11)
    chn_bare = sample(
        CT.as_turing_model(m, positives, Y), Prior(), 40; progress = false
    )
    @test vec(chn_bare[@varname(y_t[3])]) == imputed
end

@testitem "DigitalPCRError binomial link is the Poisson partition law" begin
    using EpiSewer
    using Distributions: Binomial, Poisson, clamp, pdf
    import ComposableTuringIDModels: as_turing_model, TransformObservationModel, BinomialError

    # The transform composition must yield Binomial(N, 1 - exp(-exp(Y))).
    m = EpiSewer.DigitalPCRError([1000])
    tr = EpiSewer._transformed_dpcr(m)
    @test tr isa TransformObservationModel
    @test tr.model isa BinomialError

    # Exercise the stored transform itself, checked against the Poisson
    # zero-probability the docstring claims it is (a partition tests positive
    # unless it receives no copies), computed via `Distributions.Poisson`
    # rather than a copy of the link's own formula. This would catch a wrong
    # exponent, a missing minus sign, or a swapped inner/outer `exp`.
    λ = 0.02  # copies per partition
    Y = log(λ)
    @test only(tr.transform([Y])) ≈ 1 - pdf(Poisson(λ), 0)
end

@testitem "LogNormalError survives an overflowing coefficient of variation" begin
    using EpiSewer, Distributions, Random

    # The log-scale conversion squares `sd / mean`, which is exactly the
    # coefficient of variation, so it overflows to `Inf` past sqrt(floatmax)
    # even though both moments are finite and positive. A diverging sampler
    # reaches this, and sampling a `missing` observation calls `rand`, which
    # validates and would throw.
    for σ in (1.0e155, 1.0e200, floatmax(Float64))
        d = EpiSewer.observation_error(EpiSewer.LogNormalError(), 1.0e-6, σ)
        @test logpdf(d, 1.0) == -Inf
        @test isfinite(rand(Xoshiro(1), d)) || true    # must not throw
    end
    # Just below the threshold the distribution is still constructed normally.
    d = EpiSewer.observation_error(EpiSewer.LogNormalError(), 100.0, 1.0e150)
    @test logpdf(d, 100.0) < 0
end

# The default came from R's *init* for this parameter rather than its prior, so
# it was eight times too tight. Pinned here because the two numbers sit two lines
# apart in R's source and are easy to confuse again.
@testitem "LogNormalError's default cv prior is R's, not R's init" begin
    using EpiSewer
    using ComposableTuringIDModels: HalfNormal
    using Distributions: Normal, truncated, mean, std

    cv = EpiSewer.LogNormalError().cv
    @test cv isa HalfNormal
    # R: `normal(0, 1) T[0, ]` (`EpiSewer_main.stan:910`) from
    # `noise_estimate(cv_prior_sigma = 1)`.
    r = truncated(Normal(0.0, 1.0), 0.0, Inf)
    @test mean(cv) ≈ mean(r)
    @test std(cv) ≈ std(r)
end
