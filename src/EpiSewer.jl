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

export example_data, example_distributions

# Example data loader (Zurich SARS-CoV-2 wastewater data)
include("data.jl")

# Submodules for model components (to be filled in later)
include("measurements.jl")
include("sampling.jl")
include("sewage.jl")
include("shedding.jl")
include("infections.jl")
include("forecast.jl")

end # module EpiSewer
