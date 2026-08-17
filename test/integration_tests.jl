using EpiSewer
using TestItemRunner

# Integration tests: build the EpiSewer README model as a composable
# `IDModel` (ComposableTuringIDModels) and verify the pipeline runs
# end-to-end via lightweight prior (predictive) sampling.
#
# The model mirrors the EpiSewer README example:
#   Renewal (generation time, R_t ~ RandomWalk)   -> infections I_t
#   LatentDelay (incubation period)               -> symptom onsets
#   Ascertainment (per-case shed load)            -> expected load
#   LatentDelay (shedding-load PMF)               -> delayed load
#   FlowNormalize(LogNormalError())               -> observed concentrations
#
# `EpiSewer.model(...)` assembles this as `IDModel(infection_model,
# observation_model)`.

@testitem "EpiSewer.model builds a composable IDModel that samples on example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    # Parse the example concentration strings ("NA" -> missing) to Float64.
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end

    d = EpiSewer.example_data()
    conc = _parse_conc.(d.measurements.concentration)

    # Subsample for a lightweight test (each LatentDelay truncates the expected
    # series by its PMF length, so n must exceed their total).
    sub = 5:64
    n = length(sub)
    y_obs = Vector{Union{Missing, Float64}}(conc[sub])
    flow = Vector{Float64}(d.flows.flow[sub])

    Random.seed!(42)
    mdl = EpiSewer.model()

    # It is an IDModel composing a Renewal with the flow-normalized chain.
    @test mdl isa CT.IDModel
    @test mdl.infection_model isa CT.Renewal

    # Flow is data passed through the observation-data contract.
    mdl_t = CT.as_turing_model(mdl, (y = y_obs, flow = flow), n)
    chn = sample(mdl_t, Prior(), 2; progress = false)

    @test size(chn, 1) == 2
    @test n == length(y_obs)
    # The incubation (8) and shedding (38) PMFs both truncate the series.
    @test n > 8 + 38
end

@testitem "EpiSewer.model defaults build on the full example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    d = EpiSewer.example_data()
    mdl = EpiSewer.model()  # no flow argument: flow is data

    @test mdl isa CT.IDModel
    # The default observation chain is LatentDelay(Ascertainment(LatentDelay(
    # FlowNormalize(LogNormalError())))): the incubation delay outermost (it
    # transforms I_t first), then load-per-case scaling, the shedding delay, and
    # the thin flow division innermost. The thin FlowNormalize carries no flow;
    # flow reaches the model through the observation-data contract at
    # as_turing_model time.
    @test mdl.observation_model isa CT.LatentDelay
    @test mdl.observation_model.model isa CT.Ascertainment
    @test mdl.observation_model.model.model isa CT.LatentDelay
    @test mdl.observation_model.model.model.model isa EpiSewer.FlowNormalize
    @test mdl.observation_model.model.model.model.error_model isa
        EpiSewer.LogNormalError
end

@testitem "Composable model also exercises the FlowNormalize error model in isolation" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[1:10])

    # Thin wrapper: flow is data passed through the observation-data contract.
    fn = EpiSewer.FlowNormalize(CT.NormalError())
    mdl = CT.as_turing_model(fn, (y = fill(100.0, 10), flow = flow), fill(100.0, 10))
    @test mdl !== nothing
end

@testitem "Composable model also exercises full EpiSewer.model via direct evaluation" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Random

    d = EpiSewer.example_data()
    # example_data now returns a typed Union{Missing,Float64} concentration
    # column, so the series is used directly (no string parsing).
    y_obs = Vector{Union{Missing, Float64}}(d.measurements.concentration[5:64])
    flow = Vector{Float64}(d.flows.flow[5:64])

    Random.seed!(7)
    mdl = EpiSewer.model()
    mdl_t = CT.as_turing_model(mdl, (y = y_obs, flow = flow), length(y_obs))
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

    # (a) accessible as EpiSewer.model (public but not exported)
    @test Base.ispublic(EpiSewer, :model)
    @test !Base.isexported(EpiSewer, :model)
    # (c) returns an IDModel
    d = EpiSewer.example_data()
    mdl = EpiSewer.model()
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
        CT.as_turing_model(EpiSewer.model(), (y = y_obs, flow = flow), n),
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
    obs = EpiSewer.FlowNormalize(CT.NormalError())
    mdl = EpiSewer.model(infection_model = inf, observation_model = obs)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    @test mdl.observation_model === obs

    # Overriding one component works: the other keeps its default.
    mdl_override = EpiSewer.model(infection_model = inf)
    @test mdl_override isa CT.IDModel
    @test mdl_override.infection_model === inf
    @test mdl_override.observation_model isa CT.LatentDelay
    @test mdl_override.observation_model.model.model.model isa
        EpiSewer.FlowNormalize
end

@testitem "EpiSewer.model accepts a custom infection model as default-arg override" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[5:64])

    # DirectInfections instead of the default Renewal.
    inf = CT.DirectInfections(; Z = CT.RandomWalk(), initialisation = Normal())
    mdl = EpiSewer.model(infection_model = inf)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    # The default observation chain is still assembled.
    @test mdl.observation_model isa CT.LatentDelay
    @test mdl.observation_model.model.model.model isa EpiSewer.FlowNormalize
end
