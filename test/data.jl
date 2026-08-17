using EpiSewer
using TestItemRunner

@testitem "example_data loads Zurich measurements, flows, cases" begin
    using EpiSewer
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

@testitem "example measurements sparsify to Mondays and Thursdays" begin
    using EpiSewer
    using DataFrames: nrow
    using Dates: dayname
    # The EpiSewer README example fits an artificially sparse series
    # (`.resources/EpiSewer/README.Rmd`: `weekday %in% c("Monday","Thursday")`),
    # so the dense CSV is shipped and blanked at use. The withheld days stay
    # available for the plots (#20).
    d = EpiSewer.example_data()
    sparse_days = dayname.(d.measurements.date) .∈ (["Monday", "Thursday"],)
    y = ifelse.(sparse_days, d.measurements.concentration, missing)

    @test eltype(y) == Union{Missing, Float64}
    @test length(y) == nrow(d.measurements)
    @test count(sparse_days) == 34
    # One of those 34 days (2022-02-14) is itself an unobserved day.
    @test count(!ismissing, y) == 33
    @test count(!ismissing, d.measurements.concentration) == 117
end

@testitem "example data is public but not exported" begin
    using EpiSewer
    @test Base.ispublic(EpiSewer, :example_data)
    @test !Base.isexported(EpiSewer, :example_data)
    # The hand-built PMF helper is gone: distributions go straight to the
    # components, which discretise them.
    @test !isdefined(EpiSewer, :example_distributions)
end
