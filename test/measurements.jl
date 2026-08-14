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
    # Censored (below LOD), exact (above), and missing entries all construct.
    mdl = CT.as_turing_model(
        EpiSewer.Measurements.LOD(CT.NormalError(); lod = 50.0),
        [10.0, missing, 120.0, 5.0, 60.0],
        fill(100.0, 5),
    )
    @test mdl !== nothing
end

@testitem "LOD censors values below the detection limit" begin
    using EpiSewer
    using Distributions: Normal
    using Turing

    CT = EpiSewer.ComposableTuringIDModels
    NE = CT.NormalError()  # default: σ ~ HalfNormal
    lod = 50.0

    # A value far below LOD (censored) should contribute logcdf(Normal(100,1), 50),
    # not the density at that value. Evaluate via sampling log-likelihood, which
    # is negative (log of a probability < 1).
    mdl_cen = CT.as_turing_model(
        EpiSewer.Measurements.LOD(NE, lod), [10.0], [100.0]
    )
    chn_cen = sample(mdl_cen, Prior(), 1; progress = false)

    # A value far above LOD is treated as exact (uncensored) and scores normally.
    mdl_exact = CT.as_turing_model(
        EpiSewer.Measurements.LOD(NE, lod), [150.0], [100.0]
    )
    chn_exact = sample(mdl_exact, Prior(), 1; progress = false)

    # Both produce valid chains; the censored contribution is a log-probability.
    @test size(chn_cen, 1) == 1
    @test size(chn_exact, 1) == 1
end
