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
const ADAPT_DELTA = 0.9
const MAX_DEPTH = 12
const N_CHAINS = 2
n_warmup, n_samples = 400, 300
@info "Sampling" n_warmup = n_warmup n_samples = n_samples chains = N_CHAINS
fit_seconds = @elapsed chn = sample(
    mdl, NUTS(ADAPT_DELTA; max_depth = MAX_DEPTH),
    MCMCThreads(), n_samples, N_CHAINS; warmup = n_warmup, progress = true,
)
@info "Sampling wall clock" seconds = round(fit_seconds; digits = 1)

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

# Threshold for failing the script. 1.05 is the conventional bound; 1.01 is
# stricter and is what a well-behaved fit of this size should reach, but it also
# reds on a fit that is merely under-run rather than wrong, so the looser bound
# is used and the measured value is recorded either way.
const MAX_RHAT = 1.05

diag = _fit_diagnostics(chn)
@info "Convergence diagnostics" \
    max_rhat = round(diag.max_rhat; digits = 4) \
    min_ess = round(diag.min_ess; digits = 1)

# Record the diagnostics next to the plots. Previously they were computed, logged
# and discarded, so "a clean and convergent fit" had no evidence behind it
# anywhere in the repository (#16).
diagdir = joinpath("docs", "fits")
mkpath(diagdir)
open(joinpath(diagdir, "fit_diagnostics.md"), "w") do io
    println(io, "# Worked-example fit diagnostics")
    println(io)
    println(io, "Written by `examples/ww_fit_example.jl`. Regenerate with:")
    println(io)
    println(io, "    julia --project=docs --threads=2 examples/ww_fit_example.jl")
    println(io)
    println(io, "## Sampler")
    println(io)
    println(io, "- `NUTS($ADAPT_DELTA; max_depth = $MAX_DEPTH)`")
    println(io, "- $N_CHAINS chains via `MCMCThreads()`, $n_warmup warmup + $n_samples sampling")
    println(io)
    println(io, "## Data")
    println(io)
    println(io, "- window: $(first(data.measurements.date)) to $(last(data.measurements.date))")
    println(io, "- observed days: $n_observed of $(length(y_obs)) (Mondays and Thursdays only, as the R example)")
    println(io, "- infection series length `n`: $n (observations + lead-in $(EpiSewer.observation_lead_in(EpiSewer.model())))")
    println(io)
    println(io, "## Convergence")
    println(io)
    println(io, "- max split R-hat: $(round(diag.max_rhat; digits = 4)) (threshold $MAX_RHAT)")
    println(io, "- min ESS: $(round(diag.min_ess; digits = 1))")
    println(io, "- wall clock: $(round(fit_seconds; digits = 1)) s")
end
@info "Wrote diagnostics" path = joinpath(diagdir, "fit_diagnostics.md")

# --- Save ---------------------------------------------------------------------
outdir = joinpath("docs", "fits")
mkpath(outdir)
outfile = joinpath(outdir, "ww_example_chains.jls")
# Serialise the fit INPUTS alongside the chain. The plotting script has to
# rebuild the model to extract quantities the chain does not store, and it must
# use the same `(y, flow, n)` to do so — previously it re-derived them from its
# own hard-coded window and silently reconstructed on different data from the
# one the chain was fitted on. Storing them together makes agreement structural
# rather than a convention two files have to remember.
serialize(outfile, (; chn = chn, y = y_obs, flow = flow, n = n))
@info "Saved fitted chains and their inputs" path = outfile n = n

# Fail loudly, but only after saving: a chain that did not converge is still
# worth inspecting, and the plotting script refuses to run without this file, so
# a failed fit cannot silently produce the plots the README presents as results.
diag.max_rhat <= MAX_RHAT || error(
    "fit did not converge: max R-hat $(round(diag.max_rhat; digits = 4)) " *
        "exceeds $MAX_RHAT. The chain and diagnostics were still written for " *
        "inspection. Increase warmup/sampling or revisit the model rather than " *
        "publishing plots from this chain."
)
