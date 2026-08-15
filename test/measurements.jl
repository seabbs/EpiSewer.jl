using EpiSewer
using TestItemRunner

@testitem "LOD constructs and wraps a NormalError" begin
    using EpiSewer

    m = EpiSewer.Measurements.LOD(
        EpiSewer.ComposableTuringIDModels.NormalError(); lod = 50.0
    )
    @test m.lod == 50.0
    @test m.error_model isa EpiSewer.ComposableTuringIDModels.NormalError
end

@testitem "LOD default convenience constructor" begin
    using EpiSewer

    m = EpiSewer.Measurements.LOD()
    @test m.lod == 0.0
end

@testitem "LOD model constructs for censored and missing data" begin
    using EpiSewer

    CT = EpiSewer.ComposableTuringIDModels
    # Censored (reported at LOD), exact (above), and missing entries all construct.
    mdl = CT.as_turing_model(
        EpiSewer.Measurements.LOD(CT.NormalError(); lod = 50.0),
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
        EpiSewer.Measurements.LOD(NE, lod), [50.0], [100.0]
    )
    chn_cen = sample(mdl_cen, Prior(), 1; progress = false)

    # A value above LOD is exact and scores the ordinary density.
    mdl_exact = CT.as_turing_model(
        EpiSewer.Measurements.LOD(NE, lod), [150.0], [100.0]
    )
    chn_exact = sample(mdl_exact, Prior(), 1; progress = false)

    # Both produce valid chains.
    @test size(chn_cen, 1) == 1
    @test size(chn_exact, 1) == 1
end

@testitem "LOD censored loglik matches logcdf of the underlying error distribution" begin
    using EpiSewer
    using Distributions: Normal, logcdf
    using ForwardDiff

    CT = EpiSewer.ComposableTuringIDModels
    m = EpiSewer.Measurements.LOD(CT.NormalError(; std = 1.0); lod = 50.0)
    obs_err = CT.observation_error(m, 100.0 + 1.0e-6)  # censored dist at the boundary
    # Data at the LOD scores logcdf(Normal(100, 1), 50).
    @test isapprox(
        Distributions.logpdf(obs_err, 50.0),
        logcdf(Normal(100.0, 1.0), 50.0);
        atol = 1.0e-9,
    )
    # The censored contribution is AD-differentiable w.r.t. the expected value.
    f(mu) = Distributions.logpdf(
        CT.observation_error(EpiSewer.Measurements.LOD(CT.NormalError(; std = 1.0); lod = 50.0), mu), 50.0
    )
    @test isfinite(ForwardDiff.derivative(f, 100.0))
end

@testitem "DigitalPCRError scores dPCR partition counts via transform composition" begin
    using EpiSewer
    using Turing, Distributions
    import ComposableTuringIDModels as CT

    m = EpiSewer.Measurements.DigitalPCRError([1000, 1000, 1000])
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
    m = EpiSewer.Measurements.DigitalPCRError([1000])
    tr = EpiSewer.Measurements._transformed_dpcr(m)
    @test tr isa TransformObservationModel
    @test tr.model isa BinomialError
    Y = log(0.02)  # log copies per partition
    p = 1.0 - exp(-exp(Y))
    @test p ≈ 0.0198013 atol = 1.0e-5
end
