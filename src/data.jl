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

"""
    example_distributions() -> NamedTuple

Discretised daily-delay PMFs for the EpiSewer example model assumptions:
`generation_dist`, `shedding_dist`, and `incubation_dist` (each indexed by
day). Built on request from the continuous distributions used in the EpiSewer
README example via CensoredDistributions' `double_interval_censored` (primary
within-day averaging plus daily interval censoring) with the tail renormalised
by `Distributions.truncated`:

  - generation: shifted Gamma mean 3, sd 2.4
  - shedding load: Gamma shape 0.929639, scale 7.241397
  - incubation: Gamma shape 8.5, scale 0.4

# Example
```@example example_distributions
d = EpiSewer.example_distributions()
length(d.generation_dist)
```
"""
function example_distributions()
    # Discretised Gamma PMF: mass of the doubly interval-censored Gamma in the
    # daily bins [k, k+1), right-truncated at maxX (truncation renormalises).
    pmf(dist; maxX) = [pdf(truncated(double_interval_censored(dist; interval = 1); upper = maxX), k) for k in 0:(maxX - 1)]
    generation_dist = pmf(Gamma(((3.0 - 1) / 2.4)^2, 2.4^2 / (3.0 - 1)); maxX = 15)
    shedding_dist = pmf(Gamma(0.929639, 7.241397); maxX = 38)
    incubation_dist = pmf(Gamma(8.5, 0.4); maxX = 8)
    return (; generation_dist, shedding_dist, incubation_dist)
end
