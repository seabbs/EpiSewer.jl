using EpiSewer
using TestItemRunner

@testitem "FlowNormalize constructs and wraps a NormalError" begin
    using EpiSewer
    import ComposableTuringIDModels

    mn = EpiSewer.Sewage.FlowNormalize(
        ComposableTuringIDModels.NormalError(), [1.0e14, 2.0e14];
        reference_flow = 1.5e14
    )
    @test mn.reference_flow == 1.5e14
    @test mn.flow == [1.0e14, 2.0e14]
    @test mn.error_model isa ComposableTuringIDModels.NormalError
end

@testitem "FlowNormalize reference_flow defaults to median flow" begin
    using EpiSewer
    import ComposableTuringIDModels

    mn = EpiSewer.Sewage.FlowNormalize(
        ComposableTuringIDModels.NormalError(), [1.0e14, 2.0e14, 3.0e14]
    )
    @test mn.reference_flow == 2.0e14
end

@testitem "FlowNormalize convenience constructor via keyword flow" begin
    using EpiSewer
    import ComposableTuringIDModels

    mn = EpiSewer.Sewage.FlowNormalize(
        ComposableTuringIDModels.NormalError(); flow = [1.0e14, 2.0e14]
    )
    @test mn.flow == [1.0e14, 2.0e14]
    @test mn.reference_flow == 1.5e14
end

@testitem "FlowNormalize model constructs for observed and missing data" begin
    using EpiSewer
    import ComposableTuringIDModels

    CT = ComposableTuringIDModels
    mn = EpiSewer.Sewage.FlowNormalize(
        CT.NormalError(); flow = [1.0e14, 2.0e14]
    )
    # Missing entry preserved through normalization and scored predictively.
    mdl = CT.as_turing_model(mn, [100.0, missing], [100.0, 100.0])
    @test mdl !== nothing
end

@testitem "FlowNormalize sampling works end-to-end" begin
    using EpiSewer
    using Turing
    import ComposableTuringIDModels

    CT = ComposableTuringIDModels
    mn = EpiSewer.Sewage.FlowNormalize(
        CT.NormalError(); flow = [1.0e14, 2.0e14]
    )
    mdl = CT.as_turing_model(mn, [100.0, missing], [100.0, 100.0])
    chn = sample(mdl, Prior(), 2; progress = false)
    @test size(chn, 1) == 2
end
