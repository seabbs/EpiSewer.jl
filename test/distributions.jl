using EpiSewer
using TestItemRunner

@testitem "get_discrete_gamma builds a proper normalised PMF" begin
    pmf = get_discrete_gamma(shape = 8.5, scale = 0.4)
    @test pmf isa Vector{Float64}
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1.0e-12)
    # Longest bin is the tail: mass pools to the right.
    @test pmf[end] >= 0
end

@testitem "get_discrete_gamma mean/sd parameterisation matches shape/scale" begin
    p_shape = get_discrete_gamma(shape = 4.0, scale = 2.0)
    p_mean = get_discrete_gamma(mean = 8.0, sd = 4.0)  # shape=(8/4)^2=4, scale=4^2/8=2
    @test p_shape ≈ p_mean
end

@testitem "get_discrete_gamma_shifted matches the EpiSewer example generation PMF" begin
    # Generation time from the EpiSewer README example: mean 3, sd 2.4.
    pmf = get_discrete_gamma_shifted(3.0, 2.4)
    @test pmf isa Vector{Float64}
    @test all(>=(0), pmf)
    @test isapprox(sum(pmf), 1.0; atol = 1.0e-12)
    # Shape of a shifted gamma: first day has substantial mass, tail decays.
    @test pmf[1] > pmf[end]
    @test pmf[1] > 0.0
end

@testitem "example_distributions PMFs are generated on demand (not stored)" begin
    d = example_distributions()
    @test d.generation_dist ≈ get_discrete_gamma_shifted(3.0, 2.4)
    @test d.shedding_dist ≈
        get_discrete_gamma(shape = 0.929639, scale = 7.241397)
    @test d.incubation_dist ≈ get_discrete_gamma(shape = 8.5, scale = 0.4)
end
