using EpiSewer
using TestItemRunner

# Integration tests: build the EpiSewer README model as a composable
# `IDModel` (ComposableTuringIDModels) and verify the pipeline runs
# end-to-end via lightweight prior (predictive) sampling.
#
# The model mirrors the EpiSewer README example:
#   Renewal (generation time, R_t ~ RandomWalk)   -> infections I_t   (core)
#   Ascertainment (per-case shed load)            -> expected load    (ecosystem)
#   LatentDelay (shedding-load PMF)               -> delayed load     (ecosystem)
#   FlowNormalize(NormalError, flow)              -> observed concerts (ours, wraps core)
#
# `ww_idmodel(...)` assembles this chain as `IDModel(infection_model,
# observation_model)`; the observation model receives `(y_t, I_t)` from the
# composite and each wrapper transforms the expected series inward.

@testitem "ww_idmodel builds a composable IDModel that samples on example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    # Parse the example concentration strings ("NA" -> missing) to Float64.
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end

    d = EpiSewer.example_data()
    conc = _parse_conc.(d.measurements.concentration)

    # Subsample for a lightweight test (LatentDelay truncates the expected
    # series by the shedding PMF length, so n must exceed it).
    sub = 5:64
    n = length(sub)
    y_obs = Vector{Union{Missing, Float64}}(conc[sub])
    flow = Vector{Float64}(d.flows.flow[sub])

    Random.seed!(42)
    mdl = EpiSewer.ww_idmodel(flow = flow)

    # It is an IDModel composing a Renewal with the flow-normalized chain.
    @test mdl isa CT.IDModel
    @test mdl.infection_model isa CT.Renewal

    mdl_t = CT.as_turing_model(mdl, y_obs, n)
    chn = sample(mdl_t, Prior(), 2; progress = false)

    @test size(chn, 1) == 2
    @test n == length(y_obs)
    @test n > length(EpiSewer.example_distributions().shedding_dist)
end

@testitem "ww_idmodel defaults build on the full example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    d = EpiSewer.example_data()
    mdl = EpiSewer.ww_idmodel()  # default flow from the full series

    @test mdl isa CT.IDModel
    # Default flow: full-length flows from the example data.
    @test length(mdl.observation_model.flow) == length(d.flows.flow)
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

@testitem "Composable model also exercises full ww_idmodel via direct evaluation" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Random

    d = EpiSewer.example_data()
    conc = [
        string(v) == "NA" ? missing : parse(Float64, string(v))
            for v in d.measurements.concentration[5:64]
    ]
    y_obs = Vector{Union{Missing, Float64}}(conc)
    flow = Vector{Float64}(d.flows.flow[5:64])

    Random.seed!(7)
    mdl = EpiSewer.ww_idmodel(flow = flow)
    mdl_t = CT.as_turing_model(mdl, y_obs, length(y_obs))
    res = mdl_t()

    @test haskey(res, :generated_y_t)
    @test haskey(res, :expected_y_t)
    @test haskey(res, :I_t)
    @test haskey(res, :Z_t)
end

@testitem "EpiSewer.model is public but not exported and returns an IDModel" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    # (a) accessible as EpiSewer.model
    @test isdefined(EpiSewer, :model)
    # (b) NOT exported
    @test !(:model in names(EpiSewer))
    # (c) returns an IDModel
    d = EpiSewer.example_data()
    mdl = EpiSewer.model(flow = Vector{Float64}(d.flows.flow[5:64]))
    @test mdl isa CT.IDModel
    @test mdl.infection_model isa CT.Renewal

    # (d) samples via Prior on a subsample of example data
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end
    sub = 5:64
    n = length(sub)
    y_obs = Vector{Union{Missing, Float64}}(_parse_conc.(d.measurements.concentration[sub]))
    flow = Vector{Float64}(d.flows.flow[sub])
    Random.seed!(42)
    chn = sample(
        CT.as_turing_model(EpiSewer.model(flow = flow), y_obs, n),
        Prior(), 2; progress = false
    )
    @test size(chn, 1) == 2
end

@testitem "EpiSewer.model accepts explicit composable components" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[5:64])

    inf = CT.Renewal(;
        generation_time = fill(0.25, 4), rt = CT.RandomWalk(),
        initialisation = Normal(),
    )
    obs = EpiSewer.Sewage.FlowNormalize(CT.NormalError(); flow = flow)
    mdl = EpiSewer.model(infection_model = inf, observation_model = obs)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    @test mdl.observation_model === obs

    # Overriding one component works: the other keeps its default.
    mdl_override = EpiSewer.model(infection_model = inf)
    @test mdl_override isa CT.IDModel
    @test mdl_override.infection_model === inf
    @test mdl_override.observation_model isa EpiSewer.Sewage.FlowNormalize
end

@testitem "EpiSewer.model accepts a custom infection model as default-arg override" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[5:64])

    # DirectInfections instead of the default Renewal.
    inf = CT.DirectInfections(; Z = CT.RandomWalk(), initialisation = Normal())
    mdl = EpiSewer.model(infection_model = inf, flow = flow)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    # The default observation chain is still assembled.
    @test mdl.observation_model isa EpiSewer.Sewage.FlowNormalize
end
