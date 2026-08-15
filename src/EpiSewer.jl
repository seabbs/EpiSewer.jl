module EpiSewer

# This package is a Julia port of the EpiSewer R package by Adrian Lison.
# Original: https://github.com/adrian-lison/EpiSewer
# Paper: https://doi.org/10.1038/s41467-026-75380-3

using AlgebraOfGraphics: AlgebraOfGraphics
using CSV: CSV
using CensoredDistributions: CensoredDistributions, double_interval_censored
using ComposableTuringIDModels: ComposableTuringIDModels, Renewal, RandomWalk,
    LatentDelay, Ascertainment, IDModel
using DataFrames: DataFrames, DataFrame
using DataFramesMeta: DataFramesMeta
using Distributions: Distributions, Gamma, Normal, pdf, truncated
using EpiAwareADTools: EpiAwareADTools
using PairPlots: PairPlots

# Example data loader (Zurich SARS-CoV-2 wastewater data)
include("data.jl")

# Submodules for model components (forecast/infections components not yet
# implemented; re-add their includes when the modules land).
include("measurements.jl")
include("sampling.jl")
include("sewage.jl")
include("shedding.jl")

# Default wastewater model assembly (the README example as a composable IDModel)
include("models.jl")

end # module EpiSewer
