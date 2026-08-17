using EpiSewer
using TestItemRunner

@testitem "MeasurementOutliers defaults to the EpiSewer GEV spike prior" begin
    using EpiSewer
    using Distributions: GeneralizedExtremeValue, params, quantile
    import ComposableTuringIDModels as CT

    m = EpiSewer.MeasurementOutliers(CT.NormalError())
    @test m.model isa CT.NormalError
    @test m.spike isa GeneralizedExtremeValue
    # R: outliers_estimate(gev_prior_mu = 0, sigma = 2e-8, xi = 4).
    @test params(m.spike) == (0.0, 2.0e-8, 4.0)
    @test m.scale == 1.0

    # The extreme right tail is the point: a typical day is untouched while the
    # 99% quantile is below one case-equivalent of load.
    @test quantile(m.spike, 0.5) < 1.0e-7
    @test 0.1 < quantile(m.spike, 0.99) < 1.0
end

@testitem "MeasurementOutliers composes an Ascertainment over IID" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    m = EpiSewer.MeasurementOutliers(CT.NormalError(); scale = 10.0)
    sp = EpiSewer._spiked(m)
    @test sp isa CT.Ascertainment
    @test sp.model === m.model
    @test sp.latent_prefix == "outliers"
    # The prior is the IID spike process, namespaced under `outliers`.
    @test sp.latent_model isa CT.PrefixLatentModel
    @test sp.latent_model.model isa CT.IID
    @test sp.latent_model.model.ϵ_t === m.spike

    # The transform is additive in the spike, not multiplicative.
    Y = [100.0, 200.0]
    eps = [0.0, 0.5]
    @test sp.transform(Y, eps) == Y .+ 10.0 .* eps
end

@testitem "MeasurementOutliers shifts the expected series additively" begin
    using EpiSewer
    using Turing: fix, @varname
    import ComposableTuringIDModels as CT

    Y = [100.0, 200.0, 300.0, 400.0]
    eps = [0.0, 0.25, 1.0, 0.5]
    y = [100.0, missing, 500.0, 120.0]

    function expected_with(scale)
        m = EpiSewer.MeasurementOutliers(CT.NormalError(); scale = scale)
        mdl = CT.as_turing_model(m, y, Y)
        return fix(mdl, (@varname(outliers.ϵ_t)) => eps)().expected
    end

    # Stan: kappa += load_mean * epsilon / flow_median, an additive component.
    @test expected_with(10.0) ≈ Y .+ 10.0 .* eps
    # Not multiplicative: a zero spike leaves the series alone rather than
    # zeroing it, and a zero scale is a no-op.
    @test expected_with(10.0)[1] ≈ Y[1]
    @test expected_with(0.0) ≈ Y
end

@testitem "MeasurementOutliers samples with a missing observation" begin
    using EpiSewer
    using Turing
    import ComposableTuringIDModels as CT
    import Turing.DynamicPPL as DPPL

    mdl = CT.as_turing_model(
        EpiSewer.MeasurementOutliers(CT.NormalError(); scale = 10.0),
        [100.0, missing, 500.0],
        fill(100.0, 3),
    )
    # The spike process is namespaced under `outliers`; the `missing` entry is
    # imputed by the wrapped error model rather than scored.
    vns = string.(keys(DPPL.VarInfo(mdl)))
    @test "outliers.ϵ_t" in vns
    @test "y_t[2]" in vns

    chn = sample(mdl, Prior(), 2; progress = false)
    @test size(chn, 1) == 2
end
