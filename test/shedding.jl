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
