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

Discretised PMFs for the EpiSewer example model assumptions, generated on
request from the continuous distributions used in the EpiSewer README example
(via `CensoredDistributions`/`get_discrete_gamma*`): `generation_dist`,
`shedding_dist`, and `incubation_dist`, each indexed by day. The shedding load
distribution is relative to symptom onset.

The values reproduce the discretisations shipped with the EpiSewer R package
example (generation: shifted Gamma mean 3, sd 2.4; shedding load: Gamma shape
0.929639, scale 7.241397; incubation: Gamma shape 8.5, scale 0.4).
"""
function example_distributions()
    return (
        generation_dist = get_discrete_gamma_shifted(3.0, 2.4; maxX = 15),
        shedding_dist = get_discrete_gamma(
            shape = 0.929639, scale = 7.241397; maxX = 38
        ),
        incubation_dist = get_discrete_gamma(shape = 8.5, scale = 0.4; maxX = 8),
    )
end
