module EpiSewer

# This package is a Julia port of the EpiSewer R package by Adrian Lison.
# Original: https://github.com/adrian-lison/EpiSewer
# Paper: https://doi.org/10.1038/s41467-026-75380-3

using ComposableTuringIDModels
using CensoredDistributions
using EpiAwareADTools
using DataFramesMeta
using AlgebraOfGraphics
using PairPlots
using CSV
using DataFrames

export example_data, example_distributions,
    get_discrete_gamma, get_discrete_gamma_shifted

# Discretised distribution helpers (get_discrete_gamma, get_discrete_gamma_shifted)
include("distributions.jl")

# Example data loader (Zurich SARS-CoV-2 wastewater data)
include("data.jl")

# Submodules for model components (forecast/infections components not yet
# implemented; re-add their includes when the modules land).
include("measurements.jl")
include("sampling.jl")
include("sewage.jl")
include("shedding.jl")

end # module EpiSewer
