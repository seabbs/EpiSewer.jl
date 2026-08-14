# Zurich SARS-CoV-2 example data loader.
#
# Ships the wastewater example data from the EpiSewer R package
# (see `.resources/EpiSewer/` for the original source, and
# `data/raw/extract_zurich.R` for how the CSVs were produced) so the README
# modelling example can be reproduced from Julia.
#
# The data are original EpiSewer R package example data (Zurich, provided by
# EAWAG, CC BY 4.0 license). `example_data()` returns a NamedTuple of
# DataFrames:
#   - measurements: date, concentration (gc/mL)
#   - flows:        date, flow (mL/day)
#   - cases:        date, cases (cases/day)
# and `example_distributions()` returns the discretised PMFs for the model
# assumptions as plain vectors:
#   - generation_dist, shedding_dist, incubation_dist (days since onset)

using CSV
using DataFrames

const _EXAMPLE_DATA_DIR = joinpath(@__DIR__, "..", "data", "example_zurich")

"""
    example_data() -> NamedTuple

Load the Zurich SARS-CoV-2 wastewater example data into a NamedTuple of
DataFrames (`measurements`, `flows`, `cases`). The `date` column is a `String`
in ISO `YYYY-MM-DD` format and `concentration` may contain `missing` values
for unobserved days.
"""
function example_data()
    measurements = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "measurements.csv"), DataFrame
    )
    flows = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "flows.csv"), DataFrame
    )
    cases = CSV.read(
        joinpath(_EXAMPLE_DATA_DIR, "cases.csv"), DataFrame
    )
    return (
        measurements = measurements,
        flows = flows,
        cases = cases,
    )
end

"""
    example_distributions() -> NamedTuple

Load the discretised PMFs for the EpiSewer example model assumptions as plain
`Vector{Float64}`s: `generation_dist`, `shedding_dist`, and `incubation_dist`,
each indexed by day. The shedding load distribution is relative to symptom
onset.
"""
function example_distributions()
    function _pmf(name)
        df = CSV.read(joinpath(_EXAMPLE_DATA_DIR, name), DataFrame)
        return Vector{Float64}(df.prob)
    end
    return (
        generation_dist = _pmf("generation_dist.csv"),
        shedding_dist = _pmf("shedding_dist.csv"),
        incubation_dist = _pmf("incubation_dist.csv"),
    )
end
