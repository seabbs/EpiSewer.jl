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
    using Statistics: mean

    # R's `initial_cases_crude` (`R/sewer_modeldata.R`):
    # 0.1 + mean(concentration over the window) * mean(flow) / load per case.
    # Derived here from the same conc/flow arrays the call uses, rather than
    # a value worked out by hand and copied in, so editing the inputs above
    # can't silently drift out of sync with the numbers checked below.
    conc = [100.0, 200.0, 300.0]
    flow = [1.0e11, 1.0e11, 1.0e11]
    lpc = 2.0e11
    got = EpiSewer.crude_initial_infections(conc, flow, lpc; days = 3)
    @test got ≈ 0.1 + mean(conc) * mean(flow) / lpc
    # `missing` measurements are skipped rather than propagated.
    withmissing = [100.0, missing, 300.0]
    @test EpiSewer.crude_initial_infections(
        withmissing, flow, lpc; days = 3
    ) ≈ 0.1 + mean(skipmissing(withmissing)) * mean(flow) / lpc
    @test_throws ArgumentError EpiSewer.crude_initial_infections(
        [missing, missing], flow, lpc; days = 2
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
    # At the centre of the latent path it gives R_t just above 1, as R does:
    # z = 0 leaves only the fixed intercept, so this is exactly
    # log(1 + exp(4)) / 4 — `sp(1.0, 4.0)` above, spelled out because it is
    # the one point on the curve worth naming.
    @test exp(only(EpiSewer.softplus_link([0.0]))) ≈ log1p(exp(4.0)) / 4.0
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
    using EpiSewer, Distributions, ReparameterisedDistributions

    # `_draw` short-circuits the generic construct-then-quantile path for the
    # two closed-form families. It exists for speed, so it must agree exactly.
    for m in (1.0e-3, 1.0, 500.0, 1.0e5), s in (0.1, 1.0, 50.0), z in (-3.0, 0.0, 2.5)
        s >= m && continue
        # The generic path spelled out: reparameterise onto the moments, then
        # invert through the normal CDF.
        p = cdf(Normal(), z)
        @test EpiSewer._draw(LogNormal, m, s, z) ≈ quantile(
            reparameterise(LogNormal; mean = m, sd = s, check_args = false), p
        ) rtol = 1.0e-12
        # `Normal` is already native in (mean, sd), so it has no registered
        # reparameterisation; the closed form is checked against the native
        # distribution's own inverse instead.
        @test EpiSewer._draw(Normal, m, s, z) ≈ quantile(Normal(m, s), p) rtol =
            1.0e-12
    end
end

# The renewal scan runs this once per timestep inside the differentiated
# log-density, so an abstract return type here is paid on every gradient.
# Storing the noise family as `LogNormal` rather than an instance is the easy
# way to reintroduce it: `typeof(LogNormal)` is a `UnionAll`, which says
# nothing about which family it is, so `_draw` cannot be resolved until run
# time and the incidence comes back as `Any` through the whole recursion.
@testitem "the renewal scan step infers concretely" begin
    using EpiSewer
    using Distributions: Normal, LogNormal

    for family in (LogNormal, Normal)
        noise = EpiSewer.InfectionNoise(; dist = family)
        # The family must reach the field as something dispatch can resolve
        # on, not as a `UnionAll`. `isconcretetype` is the wrong predicate
        # here: it is false for every `Type{...}`, which is exactly what we
        # want the field to be. `isdispatchelem` asks the question that
        # matters — can a method be selected on this at compile time.
        ft = fieldtype(typeof(noise), :dist)
        @test Base.isdispatchelem(ft)
        @test !(ft isa UnionAll)

        draws = EpiSewer.InfectionNoiseDraws(
            randn(8), noise.dist, noise.overdispersion,
            noise.cv_cap, noise.cv_sharpness,
        )
        rt = only(
            Base.return_types(
                EpiSewer.apply_modifier, (typeof(draws), Float64, Int)
            )
        )
        @test isconcretetype(rt)
        @test rt === Tuple{Float64, Int}
    end
end

# `SeedingRandomWalk` is checked against R's seeding block:
# `iota[1:(G+se)] = exp(random_walk([iota_log_seed_intercept]',
# iota_log_ar_noise, 0))`
# with `random_walk(s, i, 0) = cumulative_sum(append_row(s[1], i))`, the
# truncated-normal step-size prior, and the non-centred draw R's
# `vector<multiplier=iota_log_seed_sd[1]>` declaration expresses.

@testitem "SeedingRandomWalk matches R's cumulative-sum seeding" begin
    using EpiSewer
    using ComposableTuringIDModels: renewal_init_window
    using LinearAlgebra: I

    rev = reverse([0.1, 0.2, 0.3, 0.25, 0.15])
    innov = [0.1, -0.2, 0.3, 0.05]
    draws = EpiSewer.SeedingRandomWalkDraws(rev, I, innov)
    I₀ = 42.0
    # `r`, the growth rate implied by R₀, is ignored, so the window must not
    # move with it: the walk replaces the exponential, as it does in R.
    for r in (0.0, 0.7, -0.3)
        window = renewal_init_window(draws, I₀, r, 5)
        @test window ≈ I₀ .* exp.(cumsum(vcat(0.0, innov)))
        @test length(window) == 5
        @test all(window .> 0)
        # R's intercept is `iota[1]`: the EARLIEST seeded day, not the newest.
        @test window[1] == I₀
    end
end

@testitem "SeedingRandomWalk with no innovation is a flat window" begin
    using EpiSewer
    using ComposableTuringIDModels: as_turing_model, renewal_init_window
    using ComposableTuringIDModels: ConstantRenewalStep

    # A random walk with a zero step size is a flat line, whatever standard
    # normals were drawn, so the window collapses onto the intercept.
    rev = reverse([0.1, 0.2, 0.3, 0.25, 0.15])
    core = EpiSewer._seed_with(
        EpiSewer.SeedingRandomWalk(; step_size = 0.0), ConstantRenewalStep(rev)
    )
    draws = as_turing_model(core, 30)()
    @test all(iszero, draws.innovations)
    @test renewal_init_window(draws, 42.0, 0.7, 5) == fill(42.0, 5)
end

@testitem "SeedingRandomWalk takes R's step-size prior" begin
    using EpiSewer
    using Distributions: Normal, truncated
    using ComposableTuringIDModels: as_turing_model, ConstantRenewalStep
    using Random: Xoshiro

    # `rel_change_prior_mu = 0.05`, `rel_change_prior_sigma = 0.025`, scored as
    # `iota_log_seed_sd[1] ~ normal(mu, sigma) T[0, ]`.
    @test EpiSewer.SeedingRandomWalk().step_size ==
        truncated(Normal(0.05, 0.025), 0.0, Inf)

    rev = reverse([0.1, 0.2, 0.3, 0.25, 0.15])
    slots(w) = string.(
        collect(
            keys(
                rand(
                    Xoshiro(1),
                    as_turing_model(
                        EpiSewer._seed_with(w, ConstantRenewalStep(rev)), 30
                    )
                )
            )
        )
    )
    # The walk is `len_gen_int - 1` non-centred standard normals either way.
    @test any(contains("seed_raw"), slots(EpiSewer.SeedingRandomWalk()))
    # A prior on the step size adds a slot; a fixed scalar must not, as R's
    # fixed infection-noise overdispersion does not.
    @test any(contains("seed_sd"), slots(EpiSewer.SeedingRandomWalk()))
    fixed = EpiSewer.SeedingRandomWalk(; step_size = 0.05)
    @test !any(contains("seed_sd"), slots(fixed))
end

@testitem "SeedingRandomWalk rejects the shapes it does not seed" begin
    using EpiSewer
    using ComposableTuringIDModels: renewal_init_window
    using LinearAlgebra: I

    rev = reverse([0.1, 0.2, 0.3, 0.25, 0.15])
    draws = EpiSewer.SeedingRandomWalkDraws(rev, I, zeros(4))
    # The walk spans its own generation interval, not some other one.
    @test_throws DimensionMismatch renewal_init_window(draws, 42.0, 0.0, 7)
    # One walk cannot seed several strata without silently tying them.
    @test_throws ArgumentError renewal_init_window(draws, [1.0, 2.0], 0.0, 5)
    # An unresolved core still carries its priors, so it has no walk to seed
    # from.
    spec = EpiSewer.SeedingRandomWalk()
    @test_throws ErrorException renewal_init_window(spec, 42.0, 0.0, 5)
    # And a specification does not know how long its walk is.
    @test_throws ArgumentError EpiSewer._n_seed_innovations(spec.rev_gen_int)
end

# As for `InfectionNoise`: the renewal scan runs this inside the differentiated
# log-density, so an abstract type anywhere on the resolved core is paid on
# every gradient. `isconcretetype` is the wrong predicate for a field that is a
# `Type{...}`; `isdispatchelem` asks whether a method can be selected at compile
# time, which is what matters.
@testitem "the seeding walk's resolved core infers concretely" begin
    using EpiSewer
    using ComposableTuringIDModels: renewal_init_window, renewal_init_state,
        get_state
    using LinearAlgebra: I

    rev = reverse([0.1, 0.2, 0.3, 0.25, 0.15])
    draws = EpiSewer.SeedingRandomWalkDraws(rev, I, randn(4))
    for f in fieldnames(typeof(draws))
        ft = fieldtype(typeof(draws), f)
        @test Base.isdispatchelem(ft)
        @test !(ft isa UnionAll)
    end

    wt = only(
        Base.return_types(
            renewal_init_window, (typeof(draws), Float64, Float64, Int)
        )
    )
    @test isconcretetype(wt)
    @test wt === Vector{Float64}

    state = renewal_init_state(draws, 42.0, 0.7, 5)
    st = only(Base.return_types(draws, (typeof(state), Float64)))
    @test isconcretetype(st)
    @test st === typeof(state)
    @test only(
        Base.return_types(
            get_state, (typeof(draws), typeof(state), Vector{typeof(state)})
        )
    ) === Vector{Float64}
end

@testitem "model() selects the seeding walk without changing the default" begin
    using EpiSewer
    using ComposableTuringIDModels: as_turing_model, ConstantRenewalStep,
        UncertainDelay
    using Distributions: LogNormal, Normal, truncated
    using Random: Xoshiro

    a = EpiSewer.example_assumptions()
    core(; kw...) =
        EpiSewer.model(; a..., kw...).infection_model.recurrent_step.core
    # The deterministic exponential stays the default until the two seedings
    # have been compared on a fit.
    @test core() isa ConstantRenewalStep
    @test core(; seeding_walk = EpiSewer.SeedingRandomWalk()) isa
        EpiSewer.SeedingRandomWalk
    # It composes with a modifier-free renewal too, where `RenewalStep`
    # delegates straight to its core.
    @test core(;
        seeding_walk = EpiSewer.SeedingRandomWalk(), infection_noise = nothing
    ) isa EpiSewer.SeedingRandomWalk

    d = EpiSewer.example_data()
    y = d.measurements.concentration
    flow = Vector{Float64}(d.flows.flow)
    n_params(; kw...) = begin
        idm = EpiSewer.model(; a..., kw...)
        mdl = as_turing_model(
            idm, (y = y, flow = flow),
            length(y) + EpiSewer.observation_lead_in(idm)
        )
        sum(length, values(rand(Xoshiro(1), mdl)))
    end
    # `D_gen = 15` with the lag-0 bin dropped gives a 14-day seeding window, so
    # 13 innovations, plus the step size when it carries a prior.
    base = n_params()
    @test n_params(; seeding_walk = EpiSewer.SeedingRandomWalk()) == base + 14
    @test n_params(;
        seeding_walk = EpiSewer.SeedingRandomWalk(; step_size = 0.05)
    ) == base + 13

    # An inferred generation time builds its step per draw, so there is no core
    # to swap — the same restriction the infection-noise modifier has.
    gen = UncertainDelay(
        LogNormal, [Normal(1.0, 0.2), truncated(Normal(0.5, 0.2), 0, Inf)];
        D = 15.0
    )
    @test_throws ArgumentError EpiSewer.model(;
        a..., generation_time = gen,
        seeding_walk = EpiSewer.SeedingRandomWalk()
    )
end
