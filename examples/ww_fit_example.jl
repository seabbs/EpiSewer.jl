# =============================================================================
# README worked example — NUTS fit of the composable EpiSewer model.
#
# Reproduces the EpiSewer README modelling example in Julia:
#   - data: Zurich SARS-CoV-2 wastewater example (example_data())
#   - model: EpiSewer.model() — an IDModel composing
#       Renewal(generation_time, rt = RandomWalk())  (core)
#       FlowNormalize(LatentDelay(Ascertainment(NormalError, lpc), shed), flow)
#   - inference: Hamiltonian MCMC via Turing NUTS, 2 chains on 2 threads.
#
# Run from the package root:
#   julia --project=. --threads=2 examples/ww_fit_example.jl
#
# The fitted Chains are serialized to a gitignored path so the plotting
# scripts can reuse them without refitting.
# =============================================================================

using EpiSewer
using ComposableTuringIDModels: as_turing_model
using Turing
using Random, Serialization

Random.seed!(42)

# --- Data --------------------------------------------------------------------
# Concentration column has "NA" for missing values; parse to Union{Missing,Float64}.
_parse_conc(v) = let s = string(v)
    (s == "NA" || s == "missing") ? missing : parse(Float64, s)
end

data = EpiSewer.example_data()
y_obs = _parse_conc.(data.measurements.concentration)
y_obs = Vector{Union{Missing, Float64}}(y_obs)      # 120 days
flow = Vector{Float64}(data.flows.flow)            # mL/day
n = length(y_obs)

@info "Fitting composable wastewater model" n = n

# --- Model -------------------------------------------------------------------
mdl = as_turing_model(EpiSewer.model(flow = flow), y_obs, n)

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

# --- Save ---------------------------------------------------------------------
outdir = joinpath("docs", "fits")
mkpath(outdir)
outfile = joinpath(outdir, "ww_example_chains.jls")
serialize(outfile, chn)
@info "Saved fitted chains" path = outfile
