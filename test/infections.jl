using EpiSewer
using TestItemRunner

# `InfectionNoise` is checked against R's
# `approx_negative_binomial_log_noncentered`: the log-scale non-centred draw, the
# negative-binomial moments it is matched to, and the positivity that makes a
# truncation unnecessary.

@testitem "InfectionNoise matches R's log-scale non-centred draw" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels: apply_modifier, modifier_init_state

    # sigma2 = log1p(1/iota + xi^2); I = exp(log(iota) - sigma2/2 + raw*sigma)
    raw = [0.5, -1.25, 2.0]
    ξ = 0.1
    mod = EpiSewer.InfectionNoiseDraws(raw, LogNormal, ξ, Inf, 10.0)
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

@testitem "InfectionNoise caps the coefficient of variation as R does" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels: apply_modifier

    # soft_upper(x, u, k) = u - softplus(u - x, k), with softplus = log1p_exp.
    sp(x, k) = log1p(exp(k * x)) / k
    su(x, u, k) = u - sp(u - x, k)
    # Well below the cap it tracks the raw coefficient of variation closely.
    @test su(0.01, 0.5, 10.0) ≈ 0.01 atol = 2.0e-3
    @test su(0.1, 0.5, 10.0) ≈ 0.1 atol = 2.0e-2
    # Far above it, it saturates at the cap.
    @test su(50.0, 0.5, 10.0) ≈ 0.5 atol = 1.0e-6

    # The capped draw is the one the modifier uses.
    ξ, ι, raw = 0.1, 1.0e-4, 1.5
    mod = EpiSewer.InfectionNoise(; overdispersion = ξ)
    drawn = EpiSewer.InfectionNoiseDraws(
        [raw], mod.dist, ξ, mod.cv_cap, mod.cv_sharpness
    )
    got, _ = apply_modifier(drawn, ι, 0)
    σ² = log1p(su(sqrt(1 / ι + ξ^2), 0.5, 10.0)^2)
    @test got ≈ exp(log(ι) - σ² / 2 + raw * sqrt(σ²))
    # Without the cap the same tiny expectation gives a far wider draw. Compare
    # the spread rather than one draw: capping shrinks sigma, and the -sigma^2/2
    # median shift means a single capped draw can be the larger of the two.
    spread(cap) = begin
        hi, _ = apply_modifier(
            EpiSewer.InfectionNoiseDraws([2.0], LogNormal, ξ, cap, 10.0), ι, 0
        )
        lo, _ = apply_modifier(
            EpiSewer.InfectionNoiseDraws([-2.0], LogNormal, ξ, cap, 10.0), ι, 0
        )
        hi / lo
    end
    @test spread(Inf) > 1.0e4 * spread(0.5)
end

@testitem "InfectionNoise keeps infections positive" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels: apply_modifier

    # Positive by construction, so no truncation is needed and no proposal can
    # drive the expected concentration negative. An extreme draw stays positive.
    for raw in (-50.0, -8.0, 0.0, 8.0), ι in (1.0e-3, 1.0, 1.0e4)
        got, _ = apply_modifier(EpiSewer.InfectionNoiseDraws([raw], LogNormal, 0.1, Inf, 10.0), ι, 0)
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

    idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
    inf = idm.infection_model
    # R's `R_estimate_gp()` sums a short-term and a long-term Matern-3/2 GP,
    # then applies its `inv_softplus` link to the sum.
    @test inf.rt isa CT.TransformLatentModel
    combined = inf.rt.model
    @test combined isa CT.CombineLatentModels
    @test length(combined.models) == 2
    # A non-empty prefix wraps each component, which is what keeps the two GPs'
    # `σ` and `ℓ` distinct in the chain.
    @test all(m -> m isa CT.PrefixLatentModel, combined.models)
    short, long = (m.model for m in combined.models)
    @test all(g -> g isa CT.HilbertSpaceGP, (short, long))
    @test all(g -> g.kernel isa CT.Matern32Kernel, (short, long))
    @test all(g -> g.c == 3.0, (short, long))
    # Both magnitude priors, in log-R_t units and so carrying over directly.
    # The long-term term has the larger magnitude and so most of the variability.
    @test mean(short.marginal_std.untruncated) ≈ 0.125
    @test std(short.marginal_std.untruncated) ≈ 0.025
    @test mean(long.marginal_std.untruncated) ≈ 0.25
    @test std(long.marginal_std.untruncated) ≈ 0.05
    # The long-term length scale is four times the short-term one, in days.
    @test mean(long.length_scale.untruncated) ≈
        4 * mean(short.length_scale.untruncated) rtol = 1.0e-12
    # Stochastic infections, as R's `infection_noise_estimate()`.
    @test any(m -> m isa EpiSewer.InfectionNoise, inf.recurrent_step.modifiers)
    # Seeding at the scale the data implies, not at one infection.
    @test exp(mean(inf.initialisation)) > 100.0
end

@testitem "the default model draws a valid prior on the example data" begin
    using EpiSewer, Random
    import ComposableTuringIDModels as CT
    import Turing: DynamicPPL

    idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
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

@testitem "InfectionNoise takes the noise family as an argument" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels: apply_modifier

    # The machinery is generic; `dist` is what it is applied to. Both
    # location-scale families reduce to their closed forms.
    ξ, ι, z = 0.1, 400.0, 1.25
    cv = 0.5 - log1p(exp(10.0 * (0.5 - sqrt(1 / ι + ξ^2)))) / 10.0

    normal = EpiSewer.InfectionNoise(; dist = Normal, overdispersion = ξ)
    dn = EpiSewer.InfectionNoiseDraws(
        [z], normal.dist, ξ, normal.cv_cap, normal.cv_sharpness
    )
    got_n, _ = apply_modifier(dn, ι, 0)
    # R's linear form: I = iota + z * cv * iota.
    @test got_n ≈ ι + z * cv * ι

    lognormal = EpiSewer.InfectionNoise(; dist = LogNormal, overdispersion = ξ)
    dl = EpiSewer.InfectionNoiseDraws(
        [z], lognormal.dist, ξ, lognormal.cv_cap, lognormal.cv_sharpness
    )
    got_l, _ = apply_modifier(dl, ι, 0)
    σ² = log1p(cv^2)
    @test got_l ≈ exp(log(ι) - σ² / 2 + z * sqrt(σ²))

    # Both match the negative-binomial mean, and the positive family is the
    # default because the linear one can go below zero.
    @test EpiSewer.InfectionNoise().dist === LogNormal
    below, _ = apply_modifier(
        EpiSewer.InfectionNoiseDraws([-6.0], Normal, ξ, Inf, 10.0), 1.0, 0
    )
    @test below < 0
    above, _ = apply_modifier(
        EpiSewer.InfectionNoiseDraws([-6.0], LogNormal, ξ, Inf, 10.0), 1.0, 0
    )
    @test above > 0
end

@testitem "InfectionNoise moment-matches any reparameterised family" begin
    using EpiSewer, Distributions
    import ComposableTuringIDModels: apply_modifier

    # A family with no closed-form transformation routes through the quantile
    # fallback, and still lands inside the support with the right moments.
    ξ, ι = 0.1, 500.0
    d = EpiSewer.InfectionNoiseDraws([0.0], Gamma, ξ, 0.5, 10.0)
    got, _ = apply_modifier(d, ι, 0)
    @test got > 0
    @test isfinite(got)
    # At z = 0 the draw is the median, which for a Gamma sits below the mean.
    @test got < ι
end

@testitem "softplus_link is EpiSewer's inv_softplus link" begin
    using EpiSewer

    # R: apply_link(x, c(0, 4, ...)) = softplus(x, 4) = log1p_exp(4x)/4, applied
    # to the summed GPs plus an intercept fixed at 1.
    sp(x, k) = log1p(exp(k * x)) / k
    for z in (-0.8, -0.2, 0.0, 0.3, 1.0)
        @test exp(only(EpiSewer.softplus_link([z]))) ≈ sp(1.0 + z, 4.0)
    end
    # At the centre of the latent path it gives R_t just above 1, as R does.
    @test exp(only(EpiSewer.softplus_link([0.0]))) ≈ 1.00453 atol = 1.0e-5
    # Asymptotically linear rather than exponential, so the upper tail is far
    # tamer than `exp`. That is what keeps E[R_t] from carrying a compounding
    # upward bias through the renewal recursion.
    @test exp(only(EpiSewer.softplus_link([1.0]))) < exp(1.0)
    @test exp(only(EpiSewer.softplus_link([2.0]))) < 0.5 * exp(2.0)
    # Returned on the log scale, because `Renewal` applies its own `exp`.
    @test only(EpiSewer.softplus_link([0.0])) < 0.01
end

@testitem "the default R_t path applies the link without touching seeding" begin
    using EpiSewer, Distributions, Random
    import ComposableTuringIDModels as CT

    idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
    # The link sits inside the `rt` model, so `Renewal`'s own transformation is
    # left as `exp` and keeps seeding the initial incidence on the log scale.
    @test idm.infection_model.rt isa CT.TransformLatentModel
    @test idm.infection_model.rt.model isa CT.CombineLatentModels
    @test idm.infection_model.transformation === exp

    # Which is what keeps the seeded infections at the scale the data implies
    # rather than at the scale a softplus would give.
    d = EpiSewer.example_data()
    flow = Vector{Float64}(d.flows.flow)
    y = d.measurements.concentration
    n = length(y) + EpiSewer.observation_lead_in(idm)
    mdl = CT.as_turing_model(idm, (y = y, flow = flow), n)
    starts = [mdl(Xoshiro(s)).I_t[1] for s in 1:40]
    @test 50.0 < median(starts) < 20000.0
end

@testitem "the fast draw agrees with the generic moment-matched path" begin
    using EpiSewer, Distributions

    # `_draw` short-circuits the generic construct-then-quantile path for the
    # two closed-form families. It exists for speed, so it must agree exactly.
    for m in (1.0e-3, 1.0, 500.0, 1.0e5), s in (0.1, 1.0, 50.0), z in (-3.0, 0.0, 2.5)
        s >= m && continue
        generic_ln = EpiSewer._noncentred(
            EpiSewer._moment_match(LogNormal, m, s), z
        )
        @test EpiSewer._draw(LogNormal, m, s, z) ≈ generic_ln rtol = 1.0e-12
        generic_n = EpiSewer._noncentred(EpiSewer._moment_match(Normal, m, s), z)
        @test EpiSewer._draw(Normal, m, s, z) ≈ generic_n rtol = 1.0e-12
    end
end
