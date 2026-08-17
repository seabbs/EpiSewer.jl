using EpiSewer
using TestItemRunner

@testitem "MeasurementOutliers defaults to the EpiSewer GEV spike prior" begin
    using EpiSewer
    using Distributions: GeneralizedExtremeValue, Truncated, cdf, minimum,
        params, quantile
    import ComposableTuringIDModels as CT

    m = EpiSewer.MeasurementOutliers(CT.NormalError())
    @test m.model isa CT.NormalError
    @test m.scale == 1.0

    # Stan declares `vector<lower=0> epsilon` and scores it with an unnormalised
    # `gev_lpdf`, so the prior it samples is the GEV restricted to [0, Inf).
    @test m.spike isa Truncated
    @test m.spike.untruncated isa GeneralizedExtremeValue
    # R: outliers_estimate(gev_prior_mu = 0, sigma = 2e-8, xi = 4).
    @test params(m.spike.untruncated) == (0.0, 2.0e-8, 4.0)

    # No negative spikes: an outlier component must not reduce the expected
    # concentration. Untruncated, 36.8% of the mass would be below zero.
    @test minimum(m.spike) == 0.0
    @test cdf(m.spike.untruncated, 0.0) ≈ 0.3679 atol = 1.0e-4

    # The extreme right tail (xi = 4) is what makes sigma = 2e-8 meaningful: a
    # typical day is untouched, a rare day absorbs a few case-equivalents.
    @test quantile(m.spike, 0.5) ≈ 2.351e-7 rtol = 1.0e-3
    @test quantile(m.spike, 0.99) ≈ 3.092 rtol = 1.0e-3
end

@testitem "MeasurementOutliers composes an Ascertainment over IID" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    m = EpiSewer.MeasurementOutliers(CT.NormalError(); scale = 10.0)
    sp = m.spiked
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

    # The composition must be a stored field, not rebuilt per log-density
    # evaluation: `Ascertainment`'s constructor validates `transform` with
    # `hasmethod`, and Mooncake has no rule for the `Core._hasmethod`
    # foreigncall that lowers to.
    @test :spiked in fieldnames(typeof(m))
    mdl = CT.as_turing_model(m, [100.0, 200.0], Y)
    @test mdl.args.m.spiked === sp
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

    # Imputed, not held fixed: the entry is redrawn in every prior draw.
    chn = sample(mdl, Prior(), 4; progress = false)
    imputed = vec(chn[@varname(y_t[2])])
    @test allunique(imputed)
    @test all(isfinite, imputed)
end
