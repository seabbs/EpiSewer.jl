using EpiSewer
using TestItemRunner

# `InfectionNoise` is checked against R's
# `approx_negative_binomial_log_noncentered`: the log-scale non-centred draw, the
# negative-binomial moments it is matched to, and the positivity that makes a
# truncation unnecessary.

@testitem "InfectionNoise matches R's log-scale non-centred draw" begin
    using EpiSewer
    import ComposableTuringIDModels: apply_modifier, modifier_init_state

    # sigma2 = log1p(1/iota + xi^2); I = exp(log(iota) - sigma2/2 + raw*sigma)
    raw = [0.5, -1.25, 2.0]
    ξ = 0.1
    mod = EpiSewer.InfectionNoiseDraws(raw, ξ)
    ιs = (100.0, 2500.0, 7.0)
    # Thread the substate by folding, so each step reads its own `I_raw`.
    got = accumulate(
        (acc, i) -> apply_modifier(mod, ιs[i], acc[2]), eachindex(ιs);
        init = (0.0, modifier_init_state(mod, nothing))
    )
    for (i, ι) in enumerate(ιs)
        σ² = log1p(1 / ι + ξ^2)
        @test got[i][1] ≈ exp(log(ι) - σ² / 2 + raw[i] * sqrt(σ²))
    end
    # The substate is the step counter, so it advances once per step.
    @test last(got)[2] == length(ιs)
end

@testitem "InfectionNoise matches the negative binomial's first two moments" begin
    using EpiSewer, Distributions

    # The log-scale draw is a log-normal moment-matched to a negative binomial
    # of mean iota and overdispersion xi: mean iota, variance iota(1 + iota xi^2).
    for ι in (5.0, 250.0, 4000.0), ξ in (0.0, 0.1, 0.4)
        σ² = log1p(1 / ι + ξ^2)
        d = LogNormal(log(ι) - σ² / 2, sqrt(σ²))
        @test mean(d) ≈ ι rtol = 1.0e-10
        @test var(d) ≈ ι * (1 + ι * ξ^2) rtol = 1.0e-10
    end
end

@testitem "InfectionNoise keeps infections positive" begin
    using EpiSewer
    import ComposableTuringIDModels: apply_modifier

    # Positive by construction, so no truncation is needed and no proposal can
    # drive the expected concentration negative. An extreme draw stays positive.
    for raw in (-50.0, -8.0, 0.0, 8.0), ι in (1.0e-3, 1.0, 1.0e4)
        got, _ = apply_modifier(EpiSewer.InfectionNoiseDraws([raw], 0.1), ι, 0)
        @test got > 0
        @test isfinite(got)
    end
end

@testitem "InfectionNoise samples one standard normal per time point" begin
    using EpiSewer, Distributions, Random
    import ComposableTuringIDModels as CT

    r = CT.Renewal(
        [0.2, 0.3, 0.5], EpiSewer.InfectionNoise();
        rt = CT.RandomWalk(), initialisation = Normal()
    )
    drawn = rand(Xoshiro(1), CT.as_turing_model(r, 10))
    # Namespaced under the modifier's position, so it is recoverable.
    @test any(contains("I_raw"), string.(collect(keys(drawn))))
    # A fixed overdispersion is a constant, not a sampled parameter.
    @test !any(contains("ξ"), string.(collect(keys(drawn))))
end

@testitem "InfectionNoise makes infections stochastic" begin
    using EpiSewer, Distributions, Random
    import ComposableTuringIDModels as CT

    gen = [0.2, 0.3, 0.5]
    fixed = CT.Renewal(
        gen; rt = CT.FixedIntercept(0.0), initialisation = Normal(log(500.0), 0.0)
    )
    noisy = CT.Renewal(
        gen, EpiSewer.InfectionNoise(); rt = CT.FixedIntercept(0.0),
        initialisation = Normal(log(500.0), 0.0)
    )
    # With R_t and seeding both fixed the deterministic process is constant
    # across draws, and the noisy one is not.
    a = CT.as_turing_model(fixed, 12)(Xoshiro(1)).I_t
    b = CT.as_turing_model(fixed, 12)(Xoshiro(2)).I_t
    @test a ≈ b
    c = CT.as_turing_model(noisy, 12)(Xoshiro(1)).I_t
    e = CT.as_turing_model(noisy, 12)(Xoshiro(2)).I_t
    @test !(c ≈ e)
    @test all(>(0), c)
end

@testitem "gp_length_scale converts days onto the standardised index" begin
    using EpiSewer, Distributions

    # HilbertSpaceGP standardises the index, so a length scale in days is
    # divided by the standard deviation of 1:n, which is sqrt(n(n+1)/12).
    n = 164
    @test EpiSewer.gp_length_scale(21.0, n) ≈ 21.0 / sqrt(n * (n + 1) / 12)
    # A longer series makes the same number of days a shorter standardised scale.
    @test EpiSewer.gp_length_scale(21.0, 400) < EpiSewer.gp_length_scale(21.0, n)
    # With `sd` it returns the prior rather than the scaled value.
    prior = EpiSewer.gp_length_scale(21.0, n; sd = 3.5)
    @test prior isa Distribution
    @test minimum(prior) ≈ 0.05
end

@testitem "crude_initial_infections matches EpiSewer's initial_cases_crude" begin
    using EpiSewer

    # 0.1 + mean(concentration over the window) * mean(flow) / load per case.
    conc = [100.0, 200.0, 300.0]
    flow = [1.0e11, 1.0e11, 1.0e11]
    got = EpiSewer.crude_initial_infections(conc, flow, 2.0e11; days = 3)
    @test got ≈ 0.1 + 200.0 * 1.0e11 / 2.0e11
    # `missing` measurements are skipped rather than propagated.
    withmissing = [100.0, missing, 300.0]
    @test EpiSewer.crude_initial_infections(
        withmissing, flow, 2.0e11; days = 3
    ) ≈ 0.1 + 200.0 * 1.0e11 / 2.0e11
    @test_throws ArgumentError EpiSewer.crude_initial_infections(
        [missing, missing], flow, 2.0e11; days = 2
    )
end

@testitem "the default model is EpiSewer's default model" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels as CT

    idm = EpiSewer.model()
    inf = idm.infection_model
    # A Gaussian-process R_t, as R's `R_estimate_gp()`.
    @test inf.rt isa CT.HilbertSpaceGP
    @test inf.rt.kernel isa CT.Matern32Kernel
    @test inf.rt.c == 3.0
    # R's magnitude prior, which is in log-R_t units and so carries over.
    @test mean(inf.rt.marginal_std.untruncated) ≈ 0.125
    @test std(inf.rt.marginal_std.untruncated) ≈ 0.025
    # Stochastic infections, as R's `infection_noise_estimate()`.
    @test any(m -> m isa EpiSewer.InfectionNoise, inf.recurrent_step.modifiers)
    # Seeding at the scale the data implies, not at one infection.
    @test exp(mean(inf.initialisation)) > 100.0
end

@testitem "the default model draws a valid prior on the example data" begin
    using EpiSewer, Random
    import ComposableTuringIDModels as CT
    import Turing: DynamicPPL

    idm = EpiSewer.model()
    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow)
    y = d.measurements.concentration
    n = length(y) + EpiSewer.observation_lead_in(idm)
    mdl = CT.as_turing_model(idm, (y = y, flow = flow), n)
    # No duplicated variable names anywhere in the composed chain: the
    # Gaussian process and the measurement noise must not collide.
    @test DynamicPPL.DebugUtils.check_model(mdl; error_on_failure = false)
    drawn = mdl(Xoshiro(1))
    @test all(isfinite, drawn.I_t)
    # Seeded at the scale the measurements imply.
    @test 50.0 < drawn.I_t[1] < 20000.0
end
