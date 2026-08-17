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

@testitem "observation_lead_in sums the chain's LatentDelay lead-ins" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    # Default chain: the incubation (D = 8) and shedding (D = 38) PMFs each
    # shorten the expected series by `length(pmf) - 1`.
    @test EpiSewer.observation_lead_in(EpiSewer.model()) == (8 - 1) + (38 - 1)

    # A chain with no delay at all needs no lead-in.
    obs = EpiSewer.FlowNormalize(EpiSewer.LogNormalError())
    @test EpiSewer.observation_lead_in(obs) == 0
    @test EpiSewer.observation_lead_in(
        EpiSewer.model(observation_model = obs)
    ) == 0

    # Adding the sewer residence delay lengthens the lead-in by its PMF.
    res = EpiSewer.model(residence_dist = Gamma(2.0, 1.0), D_residence = 5.0)
    @test EpiSewer.observation_lead_in(res) == (8 - 1) + (38 - 1) + (5 - 1)

    # An UncertainDelay holds its PMF length constant via `D`/`Δd`.
    unc = CT.UncertainDelay(
        LogNormal, [Normal(1.5, 0.4), truncated(Normal(0.4, 0.2), 0, Inf)];
        D = 20.0
    )
    @test EpiSewer.observation_lead_in(
        CT.LatentDelay(CT.PoissonError(), unc)
    ) == 20 - 1

    # A time-varying delay: one PMF per time point, all the same length.
    tv = CT.LatentDelay(CT.PoissonError(), [[0.2, 0.3, 0.5] for _ in 1:5])
    @test EpiSewer.observation_lead_in(tv) == 3 - 1

    # Parallel streams score their own observations, so the longest branch sets
    # the lead-in rather than the sum.
    split = CT.Split(
        (
            a = CT.LatentDelay(CT.PoissonError(), [0.5, 0.5]),
            b = CT.LatentDelay(CT.PoissonError(), [0.2, 0.3, 0.5]),
        )
    )
    @test EpiSewer.observation_lead_in(split) == 3 - 1
end

@testitem "n = observations + lead-in scores every observation" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Turing
    using Random

    DPPL = Turing.DynamicPPL

    d = EpiSewer.example_data()
    mdl = EpiSewer.model()
    y = d.measurements.concentration
    flow = Vector{Float64}(d.flows.flow)

    # `n` is the length of the INFECTION series: the observed days plus the
    # chain's lead-in. Then the expected series comes out at `length(y)`, the
    # observation-error loop's right-alignment offset is zero, and no
    # observation is dropped (#18).
    n = length(y) + EpiSewer.observation_lead_in(mdl)
    res = CT.as_turing_model(mdl, (y = y, flow = flow), n)()
    @test length(res.expected_y_t) == length(y)
    @test length(res.generated_y_t) == length(y)
    @test length(res.I_t) == n

    # `n = length(y)` — the defect — leaves the lead-in unscored instead.
    short = CT.as_turing_model(mdl, (y = y, flow = flow), length(y))()
    @test length(short.expected_y_t) ==
        length(y) - EpiSewer.observation_lead_in(mdl)

    # The assertion that catches it: perturbing the FIRST observation must move
    # the log-joint. A dense series is used so that entry is observed rather
    # than sampled. The log-joint is evaluated at one fixed parameter draw,
    # reused across the perturbation, so only `y` differs.
    obs = Vector{Float64}(collect(skipmissing(y))[1:60])
    f = Vector{Float64}(flow[1:60])
    perturbed = copy(obs)
    perturbed[1] = 1.0e9

    function logjoint_at(y_in, n_in, vi)
        m = CT.as_turing_model(mdl, (y = y_in, flow = f), n_in)
        vi === nothing && return DPPL.VarInfo(Random.Xoshiro(42), m)
        return DPPL.logjoint(m, vi)
    end

    n_full = length(obs) + EpiSewer.observation_lead_in(mdl)
    vi = logjoint_at(obs, n_full, nothing)
    @test logjoint_at(obs, n_full, vi) != logjoint_at(perturbed, n_full, vi)

    # With the defect the same perturbation is bit-identical.
    vi_short = logjoint_at(obs, length(obs), nothing)
    @test logjoint_at(obs, length(obs), vi_short) ==
        logjoint_at(perturbed, length(obs), vi_short)
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
