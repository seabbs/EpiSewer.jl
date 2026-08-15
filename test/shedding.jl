using EpiSewer
using TestItemRunner

@testitem "LoadPerCase constructs with default positive prior" begin
    using EpiSewer
    using Distributions

    m = EpiSewer.Shedding.LoadPerCase()
    @test m.load_per_case isa Distributions.LogNormal
end

@testitem "LoadPerCase accepts a custom positive prior" begin
    using EpiSewer
    using Distributions: LogNormal

    m = EpiSewer.Shedding.LoadPerCase(load_per_case = LogNormal(0.0, 0.5))
    @test m.load_per_case isa Distributions.LogNormal
end

@testitem "LoadPerCase returns scaled expected load" begin
    using EpiSewer
    import ComposableTuringIDModels: as_turing_model

    m = EpiSewer.Shedding.LoadPerCase()
    infections = fill(100.0, 5)
    mdl = as_turing_model(m, infections)
    res = mdl()
    @test haskey(res, :expected_load)
    @test haskey(res, :load_per_case)
    @test length(res.expected_load) == 5
end

@testitem "LoadPerCase samples via Turing" begin
    using EpiSewer
    using Turing
    import ComposableTuringIDModels: as_turing_model

    m = EpiSewer.Shedding.LoadPerCase()
    mdl = as_turing_model(m, fill(100.0, 5))
    chn = sample(mdl, Prior(), 2; progress = false)
    @test size(chn, 1) == 2
end

@testitem "LoadPerCase scaling matches Ascertainment at the observation stage" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    # LoadPerCase at the latent stage: infections -> expected_load.
    mdl_lpc = CT.as_turing_model(EpiSewer.Shedding.LoadPerCase(), fill(100.0, 5))
    res_lpc = mdl_lpc()
    @test res_lpc.expected_load ≈ fill(100.0, 5) .* res_lpc.load_per_case

    # The identical transform at the observation stage is an Ascertainment
    # with a log-scale constant factor: with FixedIntercept(log 2) the
    # expected observations are exactly doubled (Y_t .* exp(x), x fixed at
    # log 2), the multiplicative (exponential-scale) transform Ascertainment
    # and LoadPerCase share.
    obs = CT.Ascertainment(CT.NormalError(), CT.FixedIntercept(log(2.0)))
    mdl_obs = CT.as_turing_model(obs, missing, fill(100.0, 5))
    res_obs = mdl_obs()
    @test res_obs.expected ≈ fill(200.0, 5)
end
