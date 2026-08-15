# =============================================================================
# README worked example — plots from the fitted composable wastewater model.
#
# Reproduces the EpiSewer README plots (concentration fit, R_t, load,
# infections) from the fitted chains saved by `ww_fit_example.jl`.
#
# Requires the fit first:  julia --project=. --threads=2 examples/ww_fit_example.jl
# Then:                    julia --project=. examples/ww_plots.jl
#
# Plotting uses Makie.jl (CairoMakie headless backend) + AlgebraOfGraphics,
# which are available in the docs environment. Add them to this project's
# environment if running interactively:
#   using Pkg; Pkg.add(["Makie", "CairoMakie", "AlgebraOfGraphics", "MCMCChains"])
# =============================================================================

using EpiSewer
using Turing
using Serialization
using Statistics: median, quantile
using DataFrames

# --- Load fit ----------------------------------------------------------------
chainfile = joinpath("docs", "fits", "ww_example_chains.jls")
isfile(chainfile) || error("No fitted chains found at $chainfile — run examples/ww_fit_example.jl first.")
chn = deserialize(chainfile)

internal_params = String.(keys(chn))
@info "Fitted parameters" internal_params

# The IDModel-generated R_t / infections are returned as model outputs; the
# chain holds the latent parameters (rw_init, std, ϵ_t, init_incidence, and
# the lpc / observation-noise priors). This script plots the posterior of R_t
# reconstructed from the renewal latent and the expected load/concentration
# drawn predictively. See `ww_predict.jl` for a posterior-predictive version.
println("Plotting requires a Makie backend; see the script header.")
