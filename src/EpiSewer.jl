module EpiSewer

# This is a Julia port of the EpiSewer R package by Adrian Lison.
# Original: https://github.com/adrian-lison/EpiSewer
# Paper: https://doi.org/10.1038/s41467-026-75380-3

using Turing: Turing
using DynamicPPL: @model
using Distributions: cdf, censored, ContinuousDistribution, Gamma,
    GeneralizedExtremeValue, LogNormal, Normal, quantile, truncated
using Dates: Date
using ComposableTuringIDModels: ComposableTuringIDModels,
    AbstractObservationErrorModel, AbstractObservationModel,
    AbstractRenewalModifier, Ascertainment, BinomialError,
    CombineLatentModels, HalfNormal, HilbertSpaceGP, IID, LatentDelay, Matern32Kernel,
    NormalError, PriorLike, Renewal, TransformLatentModel,
    TransformObservationModel,
    UncertainDelay, IDModel
import ComposableTuringIDModels: as_turing_model, as_turing_submodel,
    generate_observation_error_priors, observation_error,
    modifier_init_state, apply_modifier
using EpiAwareADTools: EpiAwareADTools
using ReparameterisedDistributions: reparameterise, to_native
using CSV: CSV
using DataFrames: DataFrames, DataFrame

# --- Data ---
include("data.jl")

# --- Model components ---
include("infections.jl")
include("measurements.jl")
include("sampling.jl")
include("sewage.jl")

# --- Model front end ---
include("models.jl")

# Public API (not exported — call via EpiSewer.model(), etc.)
public example_data, model, observation_lead_in
public gp_length_scale, crude_initial_infections, softplus_link
public LogNormalError, LOD, DigitalPCRError, MeasurementOutliers, FlowNormalize
public InfectionNoise, InfectionNoiseDraws

end
