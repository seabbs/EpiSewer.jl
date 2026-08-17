# =============================================================================
# README worked example — NUTS fit of the composable EpiSewer model.
#
# Reproduces the EpiSewer README modelling example in Julia:
#   - data: Zurich SARS-CoV-2 wastewater example (example_data())
#   - model: EpiSewer.model() — an IDModel composing
#       Renewal(generation_time, rt = RandomWalk())  (core)
#       LatentDelay(
#           Ascertainment(LatentDelay(FlowNormalize(LogNormalError), shed), lpc),
#           incubation)
#   - inference: Hamiltonian MCMC via Turing NUTS, 2 chains on 2 threads.
#
# Run from the package root, in the docs environment (the package itself does not
# depend on MCMCChains — see docs/Project.toml):
#   julia --project=docs --threads=2 examples/ww_fit_example.jl
#
# The fitted Chains are serialized to a gitignored path so the plotting
# scripts can reuse them without refitting.
# =============================================================================

using EpiSewer
using ComposableTuringIDModels: as_turing_model
using Turing
using MCMCChains
using DataFrames
using Dates: dayname
using Random, Serialization

Random.seed!(42)

# Data: example_data() already parses "NA" concentrations to missing, so the
# concentration column is Union{Missing,Float64} as loaded.
data = EpiSewer.example_data()
flow = Vector{Float64}(data.flows.flow)           # mL/day — data, not a model arg

# Artificially sparse measurements, as the EpiSewer README example does
# (`.resources/EpiSewer/README.Rmd`: `weekday %in% c("Monday","Thursday")`):
# keep only Mondays and Thursdays and blank the rest, so the fit is shown
# recovering the withheld days. The dense series stays available for the plots.
sparse_days = dayname.(data.measurements.date) .∈ (["Monday", "Thursday"],)
y_obs = ifelse.(sparse_days, data.measurements.concentration, missing)

# Each LatentDelay in the chain shortens the expected series, and the
# observation-error loop right-aligns the observations against what is left, so
# the infection series needs the chain's lead-in on top of the observed days —
# otherwise the first `observation_lead_in` observations are never scored (#18).
n = length(y_obs) + EpiSewer.observation_lead_in(EpiSewer.model())

n_observed = count(!ismissing, y_obs)
@info "Fitting composable wastewater model" n = n n_observed = n_observed

# --- Model -------------------------------------------------------------------
# The daily flow is passed through the OBSERVATION-DATA CONTRACT (y = concentrations,
# flow = flow_vector) at as_turing_model time — flow is data, not a model argument.
mdl = as_turing_model(EpiSewer.model(), (y = y_obs, flow = flow), n)

# --- NUTS ---------------------------------------------------------------------
# 2 chains, warmup + sampling; adapt_delta raised to 0.9 and a modest max_depth
# for the correlated latent process. Increase iter_warmup/iter_sampling for a
# higher-quality posterior (the README example used 4 chains; we use 2 per the
# replication plan, 2 threads).
n_warmup, n_samples = 400, 300
@info "Sampling" n_warmup = n_warmup n_samples = n_samples
chn = sample(
    mdl, NUTS(0.9; max_depth = 12),
    MCMCThreads(), n_samples, 2; warmup = n_warmup, progress = true,
)

# --- Diagnostics --------------------------------------------------------------
@info "Sampling complete" size(chn) = size(chn)
@info "Parameter count" n_parameters = length(keys(chn))

function _fit_diagnostics(chn)
    # R-hat (split Gelman-Rubin) from MCMCChains' flexi digest.
    gd = MCMCChains.gelmandiag(chn)
    gd_df = DataFrame(gd)
    max_rhat = maximum(psrf -> ismissing(psrf) ? 0.0 : psrf, gd_df.psrf)

    # Effective sample size per parameter.
    ess = MCMCChains.ess(chn)
    ess_df = DataFrame(ess)
    min_ess = minimum(st -> ismissing(st) ? 0.0 : st, ess_df.stat)

    return (; max_rhat = max_rhat, min_ess = min_ess)
end

try
    diag = _fit_diagnostics(chn)
    @info "Convergence diagnostics" \
        max_rhat = round(diag.max_rhat; digits = 3) \
        min_ess = round(diag.min_ess; digits = 1)
catch err
    @warn "Could not compute convergence diagnostics" exception = err
end

# --- Save ---------------------------------------------------------------------
outdir = joinpath("docs", "fits")
mkpath(outdir)
outfile = joinpath(outdir, "ww_example_chains.jls")
serialize(outfile, chn)
@info "Saved fitted chains" path = outfile
