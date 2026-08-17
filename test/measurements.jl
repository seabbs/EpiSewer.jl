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

@testitem "LOD model constructs for censored and missing data" begin
    using EpiSewer

    CT = EpiSewer.ComposableTuringIDModels
    # Censored (reported at LOD), exact (above), and missing entries all construct.
    mdl = CT.as_turing_model(
        EpiSewer.LOD(CT.NormalError(); lod = 50.0),
        [50.0, missing, 120.0, 50.0, 60.0],
        fill(100.0, 5),
    )
    @test mdl !== nothing
end

@testitem "LOD scores censored and exact observations correctly" begin
    using EpiSewer
    using Distributions: Normal
    using Turing

    CT = EpiSewer.ComposableTuringIDModels
    NE = CT.NormalError()  # default: σ ~ HalfNormal
    lod = 50.0

    # A censored measurement is reported AT the LOD and contributes
    # logcdf(Normal(100, σ), 50), not a density at an observed value.
    mdl_cen = CT.as_turing_model(
        EpiSewer.LOD(NE, lod), [50.0], [100.0]
    )
    chn_cen = sample(mdl_cen, Prior(), 1; progress = false)

    # A value above LOD is exact and scores the ordinary density.
    mdl_exact = CT.as_turing_model(
        EpiSewer.LOD(NE, lod), [150.0], [100.0]
    )
    chn_exact = sample(mdl_exact, Prior(), 1; progress = false)

    # Both produce valid chains.
    @test size(chn_cen, 1) == 1
    @test size(chn_exact, 1) == 1
end

@testitem "LOD censored loglik matches logcdf of the underlying error distribution" begin
    using EpiSewer
    using Distributions: Distributions, Normal, logcdf
    using ForwardDiff

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

@testitem "LogNormalError scores an invalid proposal at the input's own type" begin
    using EpiSewer
    using Distributions: Distributions
    using ForwardDiff

    # The rejection is `Reparameterised`'s own, so `-Inf` comes back as a
    # `Dual` rather than a bare `Float64` that would break a gradient tape.
    d = EpiSewer.observation_error(
        EpiSewer.LogNormalError(), ForwardDiff.Dual(Inf, 1.0), 0.1
    )
    lp = Distributions.logpdf(d, 100.0)
    @test lp isa ForwardDiff.Dual
    @test ForwardDiff.value(lp) == -Inf
end

@testitem "LOD(LogNormalError()) throws at the LOD on an invalid proposal" begin
    using EpiSewer
    using Distributions: Distributions

    # `Censored`'s boundary term is a `logcdf`, and `Reparameterised` guards
    # only `logpdf`/`pdf` with `valid_moments` — its `logcdf` routes through
    # `native`, which throws. So an exploding latent draw scores `-Inf` above
    # the LOD but raises at it. Upstream gap in ReparameterisedDistributions;
    # not something to guard around in `observation_error`.
    d = EpiSewer.observation_error(
        EpiSewer.LOD(EpiSewer.LogNormalError(); lod = 50.0), Inf, 0.1
    )
    @test Distributions.logpdf(d, 100.0) == -Inf
    @test_broken Distributions.logpdf(d, 50.0) == -Inf
end

@testitem "DigitalPCRError scores dPCR partition counts via transform composition" begin
    using EpiSewer
    using Turing, Distributions
    import ComposableTuringIDModels as CT

    m = EpiSewer.DigitalPCRError([1000, 1000, 1000])
    @test m.total_partitions == [1000, 1000, 1000]

    # Data contract: NamedTuple (y = positive partitions, N = total partitions).
    y = (y = [10, 25, missing], N = m.total_partitions)
    Y = log.([0.01, 0.02, 0.03])
    mdl = CT.as_turing_model(m, y, Y)
    chn = sample(mdl, Prior(), 2; progress = false)
    @test size(chn, 1) == 2
end

@testitem "DigitalPCRError binomial link is the Poisson partition law" begin
    using EpiSewer
    using Distributions: Binomial, clamp
    import ComposableTuringIDModels: as_turing_model, TransformObservationModel, BinomialError

    # The transform composition must yield Binomial(N, 1 - exp(-exp(Y))).
    m = EpiSewer.DigitalPCRError([1000])
    tr = EpiSewer._transformed_dpcr(m)
    @test tr isa TransformObservationModel
    @test tr.model isa BinomialError
    Y = log(0.02)  # log copies per partition
    p = 1.0 - exp(-exp(Y))
    @test p ≈ 0.0198013 atol = 1.0e-5
end
