using EpiSewer
using TestItemRunner

@testitem "get_discrete_gamma builds a normalised PMF via double interval censoring" begin
    pmf = get_discrete_gamma(shape = 8.5, scale = 0.4)
    @test pmf isa Vector{Float64}
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1.0e-6)
    # Longest bin is the tail: mass pools to the right.
    @test pmf[end] >= 0
end

@testitem "get_discrete_gamma mean/sd parameterisation matches shape/scale" begin
    p_shape = get_discrete_gamma(shape = 4.0, scale = 2.0)
    p_mean = get_discrete_gamma(mean = 8.0, sd = 4.0)  # shape=(8/4)^2=4, scale=4^2/8=2
    @test p_shape ≈ p_mean
end

@testitem "get_discrete_gamma uses double (primary + daily) interval censoring" begin
    # The statistically correct daily discretisation averages the primary
    # event over its day (Uniform(0,1)) AND daily interval-censors. The
    # within-day averaging shifts mass from day 0 into day 1 relative to the
    # plain unit-bin CDF: verify with the heavy-tailed shedding Gamma
    # (as used by example_distributions: maxX = 38).
    pmf = get_discrete_gamma(shape = 0.929639, scale = 7.241397; maxX = 38)
    # Double-interval-censored values (day-k mass in [k, k+1)). The explicit
    # primary-event averaging (Uniform(0,1)) moves mass toward later days
    # relative to plain unit-bin CDF differences: day 0 < day 1 here.
    @test pmf[1] ≈ 0.0810234 atol = 1.0e-5   # day 0
    @test pmf[2] ≈ 0.1338448 atol = 1.0e-5   # day 1 is the mode
    @test sum(pmf) ≈ 1.0 atol = 1.0e-6
end

@testitem "get_discrete_gamma_shifted matches the EpiSewer example generation PMF" begin
    # Generation time from the EpiSewer README example: mean 3, sd 2.4.
    # example_distributions uses maxX = 15, so compare at that length.
    pmf = get_discrete_gamma_shifted(3.0, 2.4; maxX = 15)
    @test pmf isa Vector{Float64}
    @test length(pmf) == 15
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1.0e-6)
    # Shape of a shifted gamma: first day has substantial mass, tail decays.
    @test pmf[1] > pmf[end]
    @test pmf[1] > 0.0
    # Double-interval-censored values at maxX = 15 (primary within-day
    # averaging; the tail pool shifts the early-day masses slightly compared
    # to a longer maxX, so test at this maxX):
    @test pmf[1] ≈ 0.2869314 atol = 1.0e-5
    @test pmf[2] ≈ 0.2828446 atol = 1.0e-5
end

@testitem "example_distributions PMFs are generated on demand (not stored)" begin
    d = example_distributions()
    # example_distributions calls the discrete-gamma helpers with explicit
    # maxX (15/38/8); compare against the same calls.
    @test d.generation_dist ≈
        get_discrete_gamma_shifted(3.0, 2.4; maxX = 15)
    @test d.shedding_dist ≈
        get_discrete_gamma(shape = 0.929639, scale = 7.241397; maxX = 38)
    @test d.incubation_dist ≈
        get_discrete_gamma(shape = 8.5, scale = 0.4; maxX = 8)
end
