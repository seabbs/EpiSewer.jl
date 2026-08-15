using EpiSewer
using TestItemRunner

# Integration tests: compose the new EpiSewer components with core
# ComposableTuringIDModels components into a working model and verify the
# pipeline runs end-to-end via lightweight prior (predictive) sampling.
#
# The chain composed here mirrors the EpiSewer README example:
#   Renewal (generation time, R_t ~ RandomWalk)   -> infections I_t   (core)
#   LoadPerCase (per-case shed load)              -> expected load    (ours)
#   FlowNormalize(NormalError, flow)              -> observed concerts (ours, wraps core)
# Seeding / shedding-delay / LOD are exercised by the components they compose
# alongside; this test keeps the chain minimal but genuinely multi-component.

@testitem "Composable model composes Renewal + LoadPerCase + FlowNormalize on example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, DynamicPPL, Random

    # Parse the example concentration strings ("NA" -> missing) to Float64.
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end

    d = EpiSewer.example_data()
    dst = EpiSewer.example_distributions()
    gen = dst.generation_dist
    shed = dst.shedding_dist
    conc = _parse_conc.(d.measurements.concentration)

    # Subsample for a lightweight test.
    sub = 5:60
    y_obs = Vector{Union{Missing, Float64}}(conc[sub])
    flow = Vector{Float64}(d.flows.flow[sub])
    n = length(y_obs)

    Random.seed!(42)
    NE = CT.NormalError(; std = CT.HalfNormal(0.1))
    fn_obs = EpiSewer.Sewage.FlowNormalize(NE; flow = flow)

    @model function ww_model(y_t, flow, gen, shed, n, fn_obs)
        # Core infection process: renewal with a random-walk R_t.
        infections ~ CT.as_turing_submodel(
            CT.Renewal(
                ; generation_time = gen,
                rt = CT.RandomWalk(), initialisation = Normal()
            ),
            n
        )
        I_t = infections.I_t

        # Our component: per-case shed load -> expected load.
        load ~ CT.as_turing_submodel(EpiSewer.Shedding.LoadPerCase(), I_t)
        expected_load = load.expected_load

        # Our component (wrapping core NormalError): flow-normalized observations.
        obs ~ CT.as_turing_submodel(fn_obs, y_t, expected_load)

        return (; I_t, expected_load, R_t = infections.Z_t)
    end

    mdl = ww_model(y_obs, flow, gen, shed, n, fn_obs)
    chn = sample(mdl, Prior(), 2; progress = false)

    @test size(chn, 1) == 2
    @test n == length(y_obs)
    @test length(gen) > 0 && length(shed) > 0
end

@testitem "Composable model also exercises the FlowNormalize error model in isolation" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[1:10])

    fn = EpiSewer.Sewage.FlowNormalize(CT.NormalError(); flow = flow)
    # A fully-observed series normalizes and constructs a model.
    mdl = CT.as_turing_model(fn, fill(100.0, 10), fill(100.0, 10))
    @test mdl !== nothing
end
