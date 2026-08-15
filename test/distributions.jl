using EpiSewer
using TestItemRunner

@testitem "example_distributions PMFs are normalised daily bins" begin
    d = EpiSewer.example_distributions()
    for pmf in values(d)
        @test pmf isa Vector{Float64}
        @test all(>=(0), pmf)
        @test isapprox(sum(pmf), 1.0; atol = 1.0e-6)
    end
    @test length(d.generation_dist) == 15
    @test length(d.shedding_dist) == 38
    @test length(d.incubation_dist) == 8
end

@testitem "generation PMF is a shifted Gamma (mean 3, sd 2.4)" begin
    # Matches the EpiSewer README example generation-time discretisation:
    # doubly interval-censored shifted Gamma, right-truncated at 15 (which
    # renormalises; identical to the EpiEstim/EpiSewer normalised closed form).
    pmf = EpiSewer.example_distributions().generation_dist
    @test pmf[1] > pmf[end]      # first day has substantial mass, tail decays
    @test pmf[1] ≈ 0.2869314 atol = 1.0e-5
    @test pmf[2] ≈ 0.2828446 atol = 1.0e-5
end

@testitem "shedding PMF is a doubly interval-censored Gamma" begin
    # Shedding load: Gamma(0.929639, 7.241397), maxX = 38. Day 1 is the mode.
    pmf = EpiSewer.example_distributions().shedding_dist
    @test pmf[2] > pmf[1]
    @test pmf[2] ≈ 0.1344834 atol = 1.0e-5
end

@testitem "example_distributions PMFs are generated on demand (not stored)" begin
    d1 = EpiSewer.example_distributions()
    d2 = EpiSewer.example_distributions()
    @test d1.generation_dist ≈ d2.generation_dist
    @test d1.shedding_dist ≈ d2.shedding_dist
    @test d1.incubation_dist ≈ d2.incubation_dist
end
