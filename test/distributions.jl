using EpiSewer
using TestItemRunner

# The delay distributions are passed to the components as continuous
# distributions and discretised there, so these tests check the PMFs the
# components actually built rather than a PMF this package assembled itself.

@testitem "generation interval matches R's shifted discretisation" begin
    using EpiSewer
    using Distributions
    import ComposableTuringIDModels as CT

    # Literal transcription of R's `get_discrete_gamma_shifted`
    # (`.resources/EpiSewer/R/utils_dists.R:202`), whose bins start at lag 1.
    function r_shifted(gamma_mean, gamma_sd, maxX)
        a = ((gamma_mean - 1) / gamma_sd)^2
        b = gamma_sd^2 / (gamma_mean - 1)
        F(k, sh) = k <= 0 ? 0.0 : cdf(Gamma(sh, b), k)
        res = [
            kk * F(kk, a) + (kk - 2) * F(kk - 2, a) -
                2 * (kk - 1) * F(kk - 1, a) +
                a * b * (2 * F(kk - 1, a + 1) - F(kk - 2, a + 1) - F(kk, a + 1))
                for kk in 1:maxX
        ]
        res = max.(0.0, res)
        return res ./ sum(res)
    end

    gi = EpiSewer.model(; EpiSewer.example_assumptions()...).infection_model.gen_int
    r14 = r_shifted(3.0, 2.4, 14)

    # `Renewal` drops the lag-0 bin, so the interval starts at lag 1 as R's does.
    @test length(gi) == 14
    @test maximum(abs.(gi .- r14)) < 1.0e-7

    # The mean generation interval is the 3 days the parameterisation asks for
    # (R: 2.939951024644756), not the 1.955 days the hand-built lag-0-indexed
    # PMF gave.
    @test sum((1:14) .* gi) ≈ 2.9399509958966226 atol = 1.0e-9

    # No same-day transmission: the shift puts no mass below lag 1, so the
    # lag-0 bin is exactly zero before `Renewal` drops it.
    shifted = Gamma(((3.0 - 1) / 2.4)^2, 2.4^2 / (3.0 - 1)) + 1
    @test first(CT._discretised_pmf(shifted; Δd = 1.0, D = 15.0)) == 0.0
    @test gi[1] ≈ 0.2873012 atol = 1.0e-6
end

@testitem "shedding and incubation PMFs are the R assumptions" begin
    using EpiSewer
    mdl = EpiSewer.model(; EpiSewer.example_assumptions()...)
    # `LatentDelay` stores the PMF reversed.
    incubation = reverse(mdl.observation_model.delay)
    shedding = reverse(mdl.observation_model.model.model.delay)

    @test length(incubation) == 8
    @test length(shedding) == 38
    for pmf in (incubation, shedding)
        @test all(>=(0), pmf)
        @test sum(pmf) ≈ 1.0 atol = 1.0e-9
    end

    # Shedding load: Gamma(0.929639, 7.241397), maxX = 38. Day 1 is the mode.
    @test shedding[2] > shedding[1]
    @test shedding[2] ≈ 0.1344834 atol = 1.0e-5
    # Incubation: Gamma(8.5, 0.4), mean 3.4 days.
    @test sum((0:7) .* incubation) ≈ 3.3846128 atol = 1.0e-6
end

@testitem "delay inputs take a distribution, a vector or a prior model" begin
    using EpiSewer
    using Distributions
    import ComposableTuringIDModels as CT

    # A continuous distribution is discretised by the component.
    from_dist = EpiSewer.model(; EpiSewer.example_assumptions()..., shedding_dist = Gamma(2.0, 1.0), D_shedding = 6.0)
    @test length(from_dist.observation_model.model.model.delay) == 6

    # An already discretised PMF is used as given (no `D` needed). `Renewal`
    # only drops the lag-0 bin when it discretises a distribution itself, so a
    # vector generation interval is taken verbatim — the caller owns the
    # no-same-day-transmission convention.
    pmf = fill(0.25, 4)
    from_vec = EpiSewer.model(; EpiSewer.example_assumptions()..., shedding_dist = pmf, generation_time = pmf)
    @test reverse(from_vec.observation_model.model.model.delay) ≈ pmf
    @test from_vec.infection_model.gen_int ≈ pmf

    # A prior model (delay parameters carry priors) is held and sampled per
    # draw, so the delay is inferred rather than fixed.
    uncertain = CT.UncertainDelay(
        Gamma, [truncated(Normal(2.0, 0.5), 0, Inf), truncated(Normal(1.0, 0.5), 0, Inf)];
        D = 6.0
    )
    # An inferred generation interval builds its renewal step per draw, and a
    # renewal modifier has to be composed onto an already discretised interval,
    # so the two cannot be combined (ComposableTuringIDModels issue #269). The
    # combination is rejected rather than silently dropping the noise.
    @test_throws ArgumentError EpiSewer.model(; EpiSewer.example_assumptions()..., generation_time = uncertain)

    from_prior = EpiSewer.model(;
        EpiSewer.example_assumptions()...,
        shedding_dist = uncertain, incubation_dist = uncertain,
        generation_time = uncertain, infection_noise = nothing,
    )
    @test from_prior.observation_model.delay === uncertain
    @test from_prior.observation_model.model.model.delay === uncertain
    @test from_prior.infection_model.gen_int === uncertain
end

@testitem "the incubation delay is the outermost observation wrapper" begin
    using EpiSewer
    import ComposableTuringIDModels as CT

    mdl = EpiSewer.model(; EpiSewer.example_assumptions()...)
    # Outermost first: the outer wrapper transforms the expected series first,
    # so infections reach symptom onset before the load scaling and the
    # shedding profile (Stan: `lambda = convolve(inc_rev, I)`).
    @test mdl.observation_model isa CT.LatentDelay
    @test length(mdl.observation_model.delay) == 8
    @test mdl.observation_model.model isa CT.Ascertainment
    @test mdl.observation_model.model.model isa CT.LatentDelay
    @test mdl.observation_model.model.model.model isa EpiSewer.FlowNormalize

    # No residence convolution by default (the `FlowNormalize` above sits
    # directly inside the shedding delay): EpiSewer's `residence_dist = c(1)` is
    # a point mass at same-day arrival, i.e. an identity convolution. Supplying
    # one inserts it between the shedding delay and the flow division, as Stan
    # does.
    with_res = EpiSewer.model(; EpiSewer.example_assumptions()..., residence_dist = [0.7, 0.3])
    inner = with_res.observation_model.model.model.model
    @test inner isa CT.LatentDelay
    @test reverse(inner.delay) ≈ [0.7, 0.3]
    @test inner.model isa EpiSewer.FlowNormalize
end

@testitem "the default chain evaluates forward, incubation delay included" begin
    using EpiSewer
    import ComposableTuringIDModels as CT
    using Random

    d = EpiSewer.example_data()
    y = d.measurements.concentration
    flow = Vector{Float64}(d.flows.flow)
    n = length(y)

    Random.seed!(11)
    with_inc = CT.as_turing_model(EpiSewer.model(; EpiSewer.example_assumptions()...), (y = y, flow = flow), n)()
    without = CT.as_turing_model(
        EpiSewer.model(; EpiSewer.example_assumptions()..., incubation_dist = nothing), (y = y, flow = flow), n
    )()

    @test all(isfinite, with_inc.expected_y_t)
    # A `LatentDelay` costs `length(pmf) - 1` points, so the 8-day incubation
    # PMF shortens the expected series by 7. Asserted as a difference rather
    # than an absolute length, which is still wrong for its own reason (#18).
    @test length(without.expected_y_t) - length(with_inc.expected_y_t) == 7
end
