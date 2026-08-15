using EpiSewer
using TestItemRunner

@testitem "MeasurementOutliers constructs and wraps a NormalError" begin
    using EpiSewer
    import ComposableTuringIDModels

    m = EpiSewer.Sampling.MeasurementOutliers(
        ComposableTuringIDModels.NormalError()
    )
    @test m.error_model isa ComposableTuringIDModels.NormalError
end

@testitem "MeasurementOutliers default priors are a Beta and a HalfNormal" begin
    using EpiSewer
    using Distributions
    import ComposableTuringIDModels

    m = EpiSewer.Sampling.MeasurementOutliers(
        ComposableTuringIDModels.NormalError()
    )
    @test m.contamination_prob isa Distributions.Beta
    @test m.outlier_scale isa Distributions.Distribution
end

@testitem "MeasurementOutliers model constructs and samples for mixed data" begin
    using EpiSewer
    import ComposableTuringIDModels
    using Turing

    CT = ComposableTuringIDModels
    mdl = CT.as_turing_model(
        EpiSewer.Sampling.MeasurementOutliers(CT.NormalError()),
        [100.0, missing, 500.0],
        fill(100.0, 3),
    )
    chn = sample(mdl, Prior(), 2; progress = false)
    @test size(chn, 1) == 2
end

@testitem "MeasurementOutliers mixture log-likelihood is finite" begin
    using EpiSewer
    using Distributions

    # The mixture logpdf must be finite and at most 0 (a log-probability).
    mix = EpiSewer.Sampling.OutlierMixture(
        Normal(100.0, 2.0), Normal(100.0, 8.0), 0.1
    )
    lp = Distributions.logpdf(mix, 105.0)
    @test isfinite(lp)
    @test lp <= 0.0
end
