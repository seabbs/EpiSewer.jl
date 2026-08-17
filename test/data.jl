using EpiSewer
using TestItemRunner

@testitem "example_data loads Zurich measurements, flows, cases" begin
    using DataFrames
    using Dates: Date
    d = EpiSewer.example_data()
    @test d isa NamedTuple
    @test (:measurements, :flows, :cases) ⊆ keys(d)
    @test nrow(d.measurements) == 120
    @test nrow(d.flows) == 120
    @test nrow(d.cases) == 120
    @test names(d.measurements) == ["date", "concentration"]
    @test names(d.flows) == ["date", "flow"]
    @test names(d.cases) == ["date", "cases"]
    @test eltype(d.measurements.concentration) == Union{Missing, Float64}
    # Dates are parsed, not left as ISO strings, so a plot gets a time axis.
    @test all(df -> eltype(df.date) == Date, (d.measurements, d.flows, d.cases))
    @test extrema(d.measurements.date) == (Date("2022-01-01"), Date("2022-04-30"))
end

@testitem "example_distributions returns discretised PMFs" begin
    dist = EpiSewer.example_distributions()
    @test (:generation_dist, :shedding_dist, :incubation_dist) ⊆ keys(dist)
    @test length(dist.generation_dist) == 15
    @test length(dist.shedding_dist) == 38
    @test length(dist.incubation_dist) == 8
    @test dist.generation_dist isa Vector{Float64}
    @test all(>=(0), dist.generation_dist)
    @test all(>=(0), dist.shedding_dist)
    @test all(>=(0), dist.incubation_dist)
end

@testitem "example data and distributions are public but not exported" begin
    @test Base.ispublic(EpiSewer, :example_data)
    @test Base.ispublic(EpiSewer, :example_distributions)
    @test !Base.isexported(EpiSewer, :example_data)
    @test !Base.isexported(EpiSewer, :example_distributions)
end
