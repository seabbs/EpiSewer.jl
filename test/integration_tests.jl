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
# `EpiSewer.model(; EpiSewer.example_assumptions()..., ...)` assembles this as `IDModel(infection_model,
# observation_model)`.

@testitem "EpiSewer.model assembles the default composable chain" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    mdl = EpiSewer.model(; EpiSewer.example_assumptions()...)  # no flow argument: flow is data

    @test mdl isa CT.IDModel
    @test mdl.infection_model isa CT.Renewal
    # The default observation chain is LatentDelay(Ascertainment(LatentDelay(
    # FlowNormalize(LogNormalError())))): the incubation delay outermost (it
    # transforms I_t first), then load-per-case scaling, the shedding delay, and
    # the thin flow division innermost. The thin FlowNormalize carries no flow;
    # flow reaches the model through the observation-data contract at
    # as_turing_model time.
    @test mdl.observation_model isa CT.LatentDelay
    @test mdl.observation_model.model isa EpiSewer.LoadVariation
    @test mdl.observation_model.model.model isa CT.Ascertainment
    @test mdl.observation_model.model.model.model isa CT.LatentDelay
    @test mdl.observation_model.model.model.model.model isa EpiSewer.FlowNormalize
    @test mdl.observation_model.model.model.model.model.error_model isa
        EpiSewer.MeasurementOutliers
end

@testitem "observation_lead_in sums the chain's LatentDelay lead-ins" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    # Default chain: the incubation (D = 8) and shedding (D = 38) PMFs each
    # shorten the expected series by `length(pmf) - 1`.
    @test EpiSewer.observation_lead_in(EpiSewer.model(; EpiSewer.example_assumptions()...)) == (8 - 1) + (38 - 1)

    # A chain with no delay at all needs no lead-in.
    obs = EpiSewer.FlowNormalize(EpiSewer.LogNormalError())
    @test EpiSewer.observation_lead_in(obs) == 0
    @test EpiSewer.observation_lead_in(
        EpiSewer.model(; EpiSewer.example_assumptions()..., observation_model = obs)
    ) == 0

    # Adding the sewer residence delay lengthens the lead-in by its PMF.
    res = EpiSewer.model(; EpiSewer.example_assumptions()..., residence_dist = Gamma(2.0, 1.0), D_residence = 5.0)
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
    mdl = EpiSewer.model(; EpiSewer.example_assumptions()...)
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

@testitem "FlowNormalize divides the expected load by the flow tail" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow[1:10])

    # Thin wrapper: flow is data passed through the observation-data contract.
    # The expected series is a LOAD (gc/day) and comes out as a concentration
    # (gc/mL). It is two days shorter than the flow here, as a `LatentDelay`
    # above would leave it, so the flow is aligned to its own tail — the days
    # the expected series actually covers, not the first days of the record.
    fn = EpiSewer.FlowNormalize(CT.NormalError())
    load = fill(2.0e16, 8)
    mdl = CT.as_turing_model(fn, (y = fill(100.0, 8), flow = flow), load)

    expected = mdl().expected
    @test expected ≈ load ./ flow[3:10]
    @test !isapprox(expected, load ./ flow[1:8])
end

@testitem "EpiSewer.model is public, and prior-samples the sparse example data" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    DPPL = Turing.DynamicPPL

    # Accessible as EpiSewer.model (public but not exported).
    @test Base.ispublic(EpiSewer, :model)
    @test !Base.isexported(EpiSewer, :model)

    d = EpiSewer.example_data()
    mdl = EpiSewer.model(; EpiSewer.example_assumptions()...)
    sub = 5:64
    y_obs = Vector{Union{Missing, Float64}}(d.measurements.concentration[sub])
    flow = Vector{Float64}(d.flows.flow[sub])
    n = length(y_obs) + EpiSewer.observation_lead_in(mdl)
    mdl_t = CT.as_turing_model(mdl, (y = y_obs, flow = flow), n)

    # The window holds one unmeasured day, and it is imputed: the concentration
    # column is `Union{Missing, Float64}`, which the `IDModel` narrows into the
    # carrier whose scoring path assumes the absent entries. One `y_t` variable,
    # for that day only. With `n = length(y_obs)` the day falls in the unscored
    # lead-in and there is no `y_t` variable at all.
    y_vars = filter(
        k -> startswith(string(k), "y_t"),
        collect(keys(DPPL.VarInfo(Random.Xoshiro(1), mdl_t)))
    )
    @test count(ismissing, y_obs) == 1
    @test length(y_vars) == 1

    # Every prior draw scores the real series to a finite log-joint. A misaligned
    # delay or flow division shows up here as a `-Inf`, since `LogNormalError`
    # rejects a non-positive or non-finite expected concentration.
    Random.seed!(42)
    chn = sample(mdl_t, Prior(), 2; progress = false)
    @test all(isfinite, vec(chn[:logjoint]))
end

@testitem "EpiSewer.model accepts explicit composable components" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    inf = CT.Renewal(;
        generation_time = fill(0.25, 4), rt = CT.RandomWalk(),
        initialisation = Normal(),
    )
    obs = EpiSewer.FlowNormalize(CT.NormalError())
    mdl = EpiSewer.model(; EpiSewer.example_assumptions()..., infection_model = inf, observation_model = obs)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    @test mdl.observation_model === obs

    # Overriding one component works: the other keeps its default.
    mdl_override = EpiSewer.model(; EpiSewer.example_assumptions()..., infection_model = inf)
    @test mdl_override isa CT.IDModel
    @test mdl_override.infection_model === inf
    @test mdl_override.observation_model isa CT.LatentDelay
    @test mdl_override.observation_model.model.model.model.model isa
        EpiSewer.FlowNormalize
end

@testitem "EpiSewer.model accepts a custom infection model as default-arg override" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions

    # DirectInfections instead of the default Renewal.
    inf = CT.DirectInfections(; Z = CT.RandomWalk(), initialisation = Normal())
    mdl = EpiSewer.model(; EpiSewer.example_assumptions()..., infection_model = inf)

    @test mdl isa CT.IDModel
    @test mdl.infection_model === inf
    # The default observation chain is still assembled.
    @test mdl.observation_model isa CT.LatentDelay
    @test mdl.observation_model.model.model.model.model isa EpiSewer.FlowNormalize
end

# The remaining testitems compose one package component at a time into a small
# `ComposableTuringIDModels` model and check the composition's behaviour rather
# than the component in isolation (`test/measurements.jl` and
# `test/sampling.jl` cover that). Each is a single forward evaluation or
# log-density evaluation at one fixed prior draw — no sampling — so that the
# quantity asserted is deterministic and the file stays fast.
#
# `DPPL.get_values(vi)` is the draw in the shape `DPPL.returned` takes, which is
# how a return value and a log-density are read at the *same* parameters.

@testitem "LOD composed in an IDModel left-censors at the limit" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    DPPL = Turing.DynamicPPL

    # A real infection process feeding the censored measurement model: a random
    # walk on log concentration, as the AD scenarios use.
    lod = 50.0
    n = 8
    infection = CT.DirectInfections(;
        Z = CT.RandomWalk(), initialisation = Normal(log(100.0), 0.2)
    )
    inner = CT.NormalError(; std = CT.HalfNormal(10.0))

    # The data convention (see the `LOD` docstring): a censored measurement is
    # reported AT the limit. Three variants of one series cover the branches —
    # every day above it, two days at it, one day below it.
    above = [120.0, 90.0, 70.0, 150.0, 80.0, 65.0, 110.0, 95.0]
    at_lod = copy(above)
    at_lod[[3, 6]] .= lod
    below = copy(at_lod)
    below[3] = lod - 1.0

    censored_chain(y) = CT.as_turing_model(
        CT.IDModel(infection, EpiSewer.LOD(inner; lod = lod)), y, n
    )
    exact_chain(y) = CT.as_turing_model(CT.IDModel(infection, inner), y, n)

    # One prior draw, reused for every log-joint below, so the observation model
    # is the only thing that differs between them.
    vi = DPPL.VarInfo(Random.Xoshiro(11), censored_chain(at_lod))
    draw = DPPL.get_values(vi)
    σ = only(DPPL.getindex_internal(vi, @varname(σ)))
    # The ecosystem's observation-error loop nudges the expected series by 1e-6
    # before scoring it, so the same nudge is applied to the returned series.
    Y = DPPL.returned(censored_chain(at_lod), draw).expected_y_t .+ 1.0e-6

    # A measurement at the limit scores `logcdf`, not a density, so the entire
    # difference between the censored and uncensored chains is those two
    # boundary terms: the likelihood integrates over everything below the limit
    # instead of depending on where below it the true value sat.
    correction = sum(
        logcdf(Normal(Y[i], σ), lod) - logpdf(Normal(Y[i], σ), lod)
            for i in (3, 6)
    )
    @test DPPL.logjoint(censored_chain(at_lod), vi) -
        DPPL.logjoint(exact_chain(at_lod), vi) ≈ correction rtol = 1.0e-8
    # Not a no-op: at this draw the two boundary terms are worth ~1.8 nats.
    @test !isapprox(
        DPPL.logjoint(censored_chain(at_lod), vi),
        DPPL.logjoint(exact_chain(at_lod), vi)
    )

    # Above the limit the censored chain is the uncensored chain exactly.
    @test DPPL.logjoint(censored_chain(above), vi) ==
        DPPL.logjoint(exact_chain(above), vi)

    # A measurement reported below the limit is outside the censored support.
    @test DPPL.logjoint(censored_chain(below), vi) == -Inf
end

@testitem "DigitalPCRError composed under an Ascertainment scores partitions" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    DPPL = Turing.DynamicPPL

    # The dPCR likelihood's expected series is log copies per partition, so an
    # infection series cannot feed it directly. It is composed the way the AD
    # scenarios compose it: a latent per-day deviation added to a fixed base
    # series of log copies per partition.
    n = 6
    totals = fill(1000, n)
    base = fill(log(0.02), n)
    component = CT.Ascertainment(
        EpiSewer.DigitalPCRError(totals), CT.IID(Normal(0.0, 0.3));
        transform = (Y_t, x) -> Y_t .+ x, latent_prefix = "logcopies"
    )

    # The data carries positive counts only, with one day `missing`: the
    # component supplies `N` from its own `total_partitions`, and narrows the
    # data on the way in so the `missing` day is imputed. Data that reaches an
    # error model inside a `NamedTuple` is never a model argument DynamicPPL
    # sees, so without that narrowing the scoring loop's `~` writes its draw into
    # the caller's vector and registers no variable at all.
    positives = Union{Missing, Int}[18, 25, missing, 11, 30, 22]
    observed = [1, 2, 4, 5, 6]
    mdl = CT.as_turing_model(component, positives, base)

    vi = DPPL.VarInfo(Random.Xoshiro(3), mdl)
    draw = DPPL.get_values(vi)
    x = DPPL.getindex_internal(vi, @varname(logcopies.ϵ_t))

    # The composed expected series is the Poisson partition law applied to the
    # latent log copies per partition: p_t = 1 - exp(-exp(Y_t)).
    p = DPPL.returned(mdl, draw).expected
    @test p ≈ 1.0 .- exp.(-exp.(base .+ x)) rtol = 1.0e-10

    # The observed days are then scored `Binomial(total_partitions, p_t)`.
    @test DPPL.loglikelihood(mdl, vi) ≈ sum(
        logpdf(Binomial(totals[i], p[i]), positives[i]) for i in observed
    ) rtol = 1.0e-10

    # The `missing` day is imputed rather than scored: it is absent from the
    # likelihood above and drawn as a partition count within the totals.
    imputed = only(DPPL.getindex_internal(vi, @varname(y_t[3])))
    @test isinteger(imputed)
    @test 0 <= imputed <= totals[3]

    # Supplying the totals through the data is the same model, since the
    # component's `total_partitions` is what reaches `BinomialError` either way.
    mdl_nt = CT.as_turing_model(component, (y = positives, N = totals), base)
    @test DPPL.loglikelihood(mdl_nt, vi) == DPPL.loglikelihood(mdl, vi)
end

@testitem "MeasurementOutliers composed in the wastewater chain adds a spike" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Distributions, Turing, Random

    DPPL = Turing.DynamicPPL

    shedding = [0.5, 0.3, 0.2]
    lpc = 2.0e11                       # load shed per case, gc/case
    y = [30.0, 34.0, 28.0, 41.0, 33.0, 36.0, 31.0, 29.0, 38.0, 35.0]
    flow = [
        2.5e11, 3.0e11, 2.8e11, 3.4e11, 2.6e11,
        3.1e11, 2.9e11, 3.3e11, 3.0e11, 2.7e11,
    ]
    # The R package's `load_mean / flow_median`: one unit of spike is one
    # case-equivalent of load spread over a day's flow (~0.68 gc/mL here).
    scale = lpc / (sum(flow) / length(flow))
    # An O(1) spike prior rather than the component's default, which is
    # deliberately tiny and would leave the additive term invisible against the
    # gc/mL scale. The mechanism is what is pinned here, not the prior.
    spike = Exponential(1.0)

    infection = CT.Renewal(;
        generation_time = [0.2, 0.3, 0.5], rt = CT.RandomWalk(),
        initialisation = Normal(log(50.0), 0.2)
    )
    # Placement per the `MeasurementOutliers` docstring: immediately inside
    # `FlowNormalize`, so the spike lands on the concentration scale.
    chain(s) = CT.IDModel(
        infection,
        CT.Ascertainment(
            CT.LatentDelay(
                EpiSewer.FlowNormalize(
                    EpiSewer.MeasurementOutliers(
                        EpiSewer.LogNormalError(); spike = spike, scale = s
                    )
                ),
                shedding
            ),
            Normal(log(lpc), 0.5)
        )
    )
    # `n` is the length of the infection series: the observed days plus the
    # shedding delay's lead-in, so that every observation is scored.
    n = length(y) + EpiSewer.observation_lead_in(chain(scale))
    turing_chain(s) = CT.as_turing_model(chain(s), (y = y, flow = flow), n)

    # `scale` is a construction-time constant, so the chains below share one
    # parameter space and one prior draw.
    vi = DPPL.VarInfo(Random.Xoshiro(5), turing_chain(scale))
    draw = DPPL.get_values(vi)
    spikes = DPPL.getindex_internal(vi, @varname(outliers.ϵ_t))
    unspiked = DPPL.returned(turing_chain(0.0), draw).expected_y_t
    spiked = DPPL.returned(turing_chain(scale), draw).expected_y_t

    # Every observation is scored, against one i.i.d. spike per scored day.
    @test length(unspiked) == length(y)
    @test length(spikes) == length(y)

    # The mechanism: `scale * ε_t` is ADDED to the expected concentration. So a
    # zero scale is a no-op (a multiplicative spike would zero the series
    # instead), and the spikes are per-day rather than one shared shift.
    @test spiked ≈ unspiked .+ scale .* spikes rtol = 1.0e-12
    @test all(spiked .> unspiked)
    @test length(unique(spiked .- unspiked)) == length(y)

    # At this scale the spike is not lost in rounding, and it is the spiked
    # series that gets scored.
    @test !isapprox(spiked, unspiked; rtol = 1.0e-3)
    @test DPPL.logjoint(turing_chain(scale), vi) !=
        DPPL.logjoint(turing_chain(0.0), vi)
end
