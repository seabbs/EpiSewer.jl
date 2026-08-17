using EpiSewer
using TestItemRunner

# The composed behaviour — dividing the expected load by the flow, aligned to
# the flow's tail — is asserted in `integration_tests.jl`, by "FlowNormalize
# divides the expected load by the flow tail". What is left here is the thin
# wrapper's own contract: what it stores, and what it demands of the data.

@testitem "FlowNormalize constructs and wraps a NormalError" begin
    using EpiSewer
    import ComposableTuringIDModels

    # Thin wrapper: a single error_model field, no flow/reference_flow.
    mn = EpiSewer.FlowNormalize(ComposableTuringIDModels.NormalError())
    @test mn.error_model isa ComposableTuringIDModels.NormalError
    @test !(hasproperty(mn, :flow))
    @test !(hasproperty(mn, :reference_flow))
end

@testitem "FlowNormalize requires flow in the observation-data contract" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    mn = EpiSewer.FlowNormalize(CT.NormalError())
    # Without the `:flow` field the wrapper errors when the model is evaluated
    # (as_turing_model returns a DynamicPPL model; the assert fires on eval).
    mdl = CT.as_turing_model(mn, (y = [100.0, 100.0],), [100.0, 100.0])
    @test_throws AssertionError mdl()
end
