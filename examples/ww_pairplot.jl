# =============================================================================
# README worked example — PairsPlot of prior vs posterior densities.
#
# Shows pairs of key scalar parameters (NOT the latent time-varying ones):
# the load-per-case (lpc), the observation-noise standard deviation, and the
# random-walk R_t step standard deviation — with both a prior sample and the
# fitted posterior from `ww_fit_example.jl`.
#
# Requires:  julia --project=. --threads=2 examples/ww_fit_example.jl
# then:      julia --project=. examples/ww_pairplot.jl
# =============================================================================

using EpiSewer
using ComposableTuringIDModels: as_turing_model
using PairPlots, Distributions
using Turing
using Random, Serialization

Random.seed!(42)

# --- Data --------------------------------------------------------------------
_parse_conc(v) = let s = string(v)
    (s == "NA" || s == "missing") ? missing : parse(Float64, s)
end
data = EpiSewer.example_data()
y_obs = Vector{Union{Missing, Float64}}(_parse_conc.(data.measurements.concentration))
flow = Vector{Float64}(data.flows.flow)
n = length(y_obs)

mdl = as_turing_model(EpiSewer.model(flow = flow), y_obs, n)

# --- Prior sample of the key scalar parameters -------------------------------
n_prior = 500
chn_prior = sample(mdl, Prior(), n_prior; progress = false)

# Key scalar parameter names (verify against names(chn, :parameters)).
key_params = intersect(
    ("lpc", "σ", "rw_std", "std", "eps_latent"),
    String.(keys(chn_prior)),
)
@info "Key parameters available" key_params

# --- Posterior ----------------------------------------------------------------
chainfile = joinpath("docs", "fits", "ww_example_chains.jls")
chn_post = isfile(chainfile) ? deserialize(chainfile) : nothing

if chn_post !== nothing && !isempty(key_params)
    # Posterior as a PairPlots-friendly DataFrame.
    post_df = DataFrame(
        sym => vec(Array(chn_post[group = :parameters, sym])) for sym in key_params
    )
    prior_df = DataFrame(
        sym => vec(Array(chn_prior[group = :parameters, sym])) for sym in key_params
    )
    g = PairPlots.pairplot(
        (Prior = prior_df, Posterior = post_df);
        markersize = Dict(:Prior => 2, :Posterior => 3),
    )
    save("docs/fits/ww_pairplot_prior_posterior.png", g)
    @info "Saved pairplot" path = "docs/fits/ww_pairplot_prior_posterior.png"
else
    @warn "Posterior missing or no shared key parameters — run the fit first."
end
