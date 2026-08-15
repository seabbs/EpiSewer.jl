module EpiSewer

# This is a Julia port of the EpiSewer R package by Adrian Lison.
# Original: https://github.com/adrian-lison/EpiSewer
# Paper: https://doi.org/10.1038/s41467-026-75380-3

using Turing: Turing
using DynamicPPL: DynamicPPL, @model, to_submodel
using Distributions: Distributions, Beta, censored, ContinuousUnivariateDistribution,
    Distribution, Gamma, LogNormal, mean, Normal, pdf, truncated
using CensoredDistributions: CensoredDistributions, double_interval_censored
using ComposableTuringIDModels: ComposableTuringIDModels,
    AbstractObservationErrorModel, AbstractObservationModel,
    Ascertainment, BinomialError, HalfNormal, LatentDelay, MissingObservations,
    NormalError, PriorLike, Renewal, RandomWalk, TransformObservationModel,
    IDModel
import ComposableTuringIDModels: as_turing_model, as_turing_submodel,
    generate_observation_error_priors, observation_error, _at
using EpiAwareADTools: EpiAwareADTools
using ReparameterisedDistributions: reparameterise
using CSV: CSV
using DataFrames: DataFrames, DataFrame
using DataFramesMeta: DataFramesMeta
using AlgebraOfGraphics: AlgebraOfGraphics
using PairPlots: PairPlots


# --- Data ---
include("data.jl")

# --- Model components ---
include("measurements.jl")
include("sampling.jl")
include("sewage.jl")

# --- Model front end ---
include("models.jl")

# Public API (not exported — call via EpiSewer.model(), etc.)
public example_data, example_distributions, model
public LogNormalError, LOD, DigitalPCRError, MeasurementOutliers, FlowNormalize

end
