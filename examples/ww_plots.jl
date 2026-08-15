# =============================================================================
# README worked example — plots from the fitted composable wastewater model.
#
# Reproduces the EpiSewer README plots (R_t, and prior-vs-posterior pairs for
# the key scalar parameters) from the fitted chains saved by
# `ww_fit_example.jl`, using the docs environment (Makie + PairPlots).
#
# Requires: julia --project=docs --threads=2 examples/ww_plots.jl
# =============================================================================

using EpiSewer
using ComposableTuringIDModels: as_turing_model
using Turing, MCMCChains
using Distributions
using Random, Serialization, Statistics
using CairoMakie
using PairPlots
using DataFrames

Random.seed!(42)

outdir = "docs/fits"
mkpath(outdir)
chainfile = joinpath(outdir, "ww_example_chains.jls")

# --- Load (or run a short) fit ------------------------------------------------
function get_chain()
    if isfile(chainfile)
        return deserialize(chainfile)
    end
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end
    data = EpiSewer.example_data()
    y = Vector{Union{Missing, Float64}}(_parse_conc.(data.measurements.concentration))
    flow = Vector{Float64}(data.flows.flow)
    mdl = as_turing_model(EpiSewer.model(flow = flow), y, length(y))
    chn = sample(mdl, NUTS(0.9; max_depth = 12), MCMCThreads(), 60, 2; warmup = 60, progress = false)
    serialize(chainfile, chn)
    return chn
end

chn = get_chain()

# --- R_t over time ------------------------------------------------------------
# R_t is the exp of the renewal latent Z_t. The RandomWalk latent is
#   Z_t = rw_init + cumsum(ϵ_t)
# with rw_init and ϵ_t stored as chain parameters, so R_t is reconstructed
# directly from the fitted chain (no predict needed).
pnames = [string(k) for k in keys(chn)]
rw_init = chn[:rw_init]                          # (draws, chains) scalars
eps = chn[Symbol("ϵ_t")]                          # (draws, chains) vectors of length n-1
n_draw, n_chain = size(eps)
n = length(first(eps)) + 1                       # n-1 innovations -> n time points
function _reconstruct_R(rw_init, eps)
    n_draw, n_chain = size(eps)
    n = length(first(eps)) + 1
    R_draws = zeros(n_draw * n_chain, n)
    idx = 0
    for d in 1:n_draw, c in 1:n_chain
        idx += 1
        Z = vcat(rw_init[d, c], rw_init[d, c] .+ cumsum(eps[d, c]))
        R_draws[idx, :] .= exp.(Z)
    end
    return R_draws
end
R_draws = _reconstruct_R(rw_init, eps)
med = vec(median(R_draws; dims = 1))
lo95 = [quantile(R_draws[:, i], 0.025) for i in axes(R_draws, 2)]
hi95 = [quantile(R_draws[:, i], 0.975) for i in axes(R_draws, 2)]
t = 1:length(med)

fig = Figure(size = (900, 420))
ax = Axis(
    fig[1, 1]; title = "Effective reproduction number R_t",
    xlabel = "day", ylabel = "R_t"
)
band!(ax, t, lo95, hi95; color = (:steelblue, 0.3), label = "95% CI")
lines!(ax, t, med; color = :steelblue, label = "median")
hlines!(ax, [1.0]; color = :red, linestyle = :dash, label = "R = 1")
axislegend(ax; position = :lt)
save(joinpath(outdir, "ww_plot_Rt.png"), fig)
@info "Saved R_t plot" path = "docs/fits/ww_plot_Rt.png"

# --- Prior vs posterior PairPlots for key scalar parameters -------------------
# Key scalar params: the load-per-case, observation noise, RW step std.
key_names = filter(p -> occursin(r"^Parameter\((std|σ|lpc|rw_init|Ascertainment)", p), pnames)
key_names = map(p -> match(r"^Parameter\((.+)\)", p).captures[1], key_names)
key_names = unique(key_names)
if length(key_names) >= 2
    # Build the DataFrames column-by-column with plain String column names:
    # `DataFrame(name => col for ...)` in generator form produces a malformed
    # frame ("first"/"second" columns holding the name strings), which then
    # reaches PairPlots' kernel-density bandwidth estimator as a
    # Vector{SubString{String}} and errors.
    cols = String.(key_names)
    syms = Symbol.(cols)
    post_df = DataFrame()
    for (nm, sym) in zip(cols, syms)
        post_df[!, nm] = vec(chn[sym])
    end

    # Prior sample of the same parameters from the model.
    _parse_conc(v) = let s = string(v)
        (s == "NA" || s == "missing") ? missing : parse(Float64, s)
    end
    data = EpiSewer.example_data()
    y = Vector{Union{Missing, Float64}}(_parse_conc.(data.measurements.concentration))
    flow = Vector{Float64}(data.flows.flow)
    mdl = as_turing_model(EpiSewer.model(flow = flow), y, length(y))
    prior_chn = sample(mdl, Prior(), 500; progress = false)

    prior_df = DataFrame()
    for (nm, sym) in zip(cols, syms)
        prior_df[!, nm] = vec(prior_chn[sym])
    end

    # Distinguish the two series by colour (a series-level `markersize` would
    # be forwarded to the diagonal density Lines plots, which reject it).
    g = PairPlots.pairplot(
        PairPlots.Series(prior_df; label = "Prior", color = :grey),
        PairPlots.Series(post_df; label = "Posterior", color = :steelblue),
    )
    save(joinpath(outdir, "ww_pairplot_prior_posterior.png"), g)
    @info "Saved pairplot" keys = cols path = "docs/fits/ww_pairplot_prior_posterior.png"
else
    @warn "Not enough key scalar params for pairplot; found: $(key_names)"
end
