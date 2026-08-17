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

@testitem "example data is public but not exported" begin
    @test Base.ispublic(EpiSewer, :example_data)
    @test !Base.isexported(EpiSewer, :example_data)
    # The hand-built PMF helper is gone: distributions go straight to the
    # components, which discretise them.
    @test !isdefined(EpiSewer, :example_distributions)
end
