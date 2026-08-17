# Zurich SARS-CoV-2 example data loader.
#
# Ships the wastewater example data from the EpiSewer R package
# (see `.resources/EpiSewer/` for the original source, and
# `data/raw/extract_zurich.R` for how the CSVs were produced) so the README
# modelling example can be reproduced from Julia.

const _EXAMPLE_DATA_DIR = joinpath(@__DIR__, "..", "data", "example_zurich")

"""
    example_data() -> NamedTuple

Load the Zurich SARS-CoV-2 wastewater example data (2022-01-01..2022-04-30)
as a NamedTuple of DataFrames (`measurements`, `flows`, `cases`). Dates are
`Date`s; `concentration`/`cases` may be `missing` for unobserved days. The
model naturally accounts for missing or non-daily observations.

# Example
```@example example_data
d = EpiSewer.example_data()
size(d.measurements)
```
"""
function example_data()
    measurements = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "measurements.csv"), DataFrame;
        missingstring = "NA",
        types = Dict("date" => Date, "concentration" => Union{Missing, Float64}),
    )
    flows = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "flows.csv"), DataFrame;
        types = Dict("date" => Date, "flow" => Float64),
    )
    cases = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "cases.csv"), DataFrame;
        missingstring = "NA",
        types = Dict("date" => Date, "cases" => Union{Missing, Float64}),
    )
    return (measurements = measurements, flows = flows, cases = cases)
end
