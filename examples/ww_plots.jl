# =============================================================================
# README worked example — plots from the fitted composable wastewater model.
#
# Reproduces the EpiSewer README plots (R_t, and prior-vs-posterior pairs for
# the key scalar parameters) from the fitted chains saved by
# `ww_fit_example.jl`, using the docs environment (Makie + PairPlots).
#
# TODO(follow-up): reproduce the remaining original README plots
#   (concentration fit, expected load, infections ± cases, growth report).
#   These need predictive (`predict`) or expected-series reconstruction from
#   the chain — more machinery than the R_t / pairplot panels here; see the
#   plot list in `.resources/EpiSewer/README.Rmd`.
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
# Fixed fit window: the chain and the conditional-extraction model must be
# built on the SAME (y, flow) subsample, so the plots reconstruct quantities
# for the window the chain was fitted on.
const WINDOW = 1:55

function get_chain()
    if isfile(chainfile)
        return deserialize(chainfile)
    end
    data = EpiSewer.example_data()
    y = data.measurements.concentration[WINDOW]
    flow = Vector{Float64}(data.flows.flow[WINDOW])
    mdl = as_turing_model(EpiSewer.model(), (y = y, flow = flow), length(y))
    chn = sample(mdl, NUTS(0.9; max_depth = 12), MCMCThreads(), 60, 2; warmup = 60, progress = false)
    serialize(chainfile, chn)
    return chn
end

chn = get_chain()

# --- Chain-key guard -----------------------------------------------------------
# The chain keys follow the FlexiChains "Parameter(...)" format and the model
# internals (`rw_init`, `ϵ_t`) come from ComposableTuringIDModels' RandomWalk.
# Guard on those so a rename degrades to a clear warning rather than a crash.
function _missing_keys(required, pnames)
    miss = [k for k in required if !any(p -> p == k, pnames)]
    if !isempty(miss)
        @warn "Expected chain parameter(s) not found — plot section skipped." missing = miss actual = pnames
    end
    return isempty(miss) ? nothing : miss
end

# --- R_t over time ------------------------------------------------------------
# R_t is the exp of the renewal latent Z_t. The RandomWalk latent is
#   Z_t = rw_init + cumsum(ϵ_t)
# with rw_init and ϵ_t stored as chain parameters, so R_t is reconstructed
# directly from the fitted chain (no predict needed).
function _reconstruct_from_chain(chn)
    pnames = [string(k) for k in keys(chn)]
    _missing_keys(["Parameter(rw_init)", "Parameter(ϵ_t)"], pnames) !== nothing && return nothing
    rw_init = chn[:rw_init]                      # (draws, chains) scalars
    eps = chn[Symbol("ϵ_t")]                     # (draws, chains) vectors of length n-1
    n_draw, n_chain = size(eps)
    n = length(first(eps)) + 1                   # n-1 innovations -> n time points
    R_draws = zeros(n_draw * n_chain, n)
    idx = 0
    for d in 1:n_draw, c in 1:n_chain
        idx += 1
        Z = vcat(rw_init[d, c], rw_init[d, c] .+ cumsum(eps[d, c]))
        R_draws[idx, :] .= exp.(Z)
    end
    return R_draws
end

R_draws = _reconstruct_from_chain(chn)
if !isnothing(R_draws)
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
end

# --- Generated-quantity extraction (concentration / load / infections) ------
# The IDModel's returned quantities (generated_y_t, expected_y_t, I_t, Z_t) are
# not stored as chain parameters, so `predict` cannot surface them on this
# FlexiChains setup (it returns only the parameter keys + Extras). Instead we
# condition the model on one posterior draw's latent parameters and evaluate
# it: `DynamicPPL.condition` fixes rw_init/std/ϵ_t/init_incidence/lpc/σ and the
# model runs forward deterministically, returning the exact generated
# quantities. This reconstructs I_t / expected load / expected concentration
# from the SAME model math that produced the fit, draw by draw.
function _extract_generated(chn, mdl)
    pnames = [string(k) for k in keys(chn)]
    required = [
        "Parameter(rw_init)", "Parameter(std)", "Parameter(ϵ_t)",
        "Parameter(init_incidence)", "Parameter(Ascertainment.intercept)",
        "Parameter(σ)",
    ]
    _missing_keys(required, pnames) !== nothing && return nothing
    n_draw, n_chain = size(chn[:rw_init])
    n = length(first(chn[Symbol("ϵ_t")])) + 1

    I_draws = zeros(n_draw * n_chain, n)
    pred_draws = zeros(n_draw * n_chain, n)
    load_draws = zeros(n_draw * n_chain, n)
    lpc = vec(chn[Symbol("Ascertainment.intercept")])

    idx = 0
    for c in 1:n_chain, d in 1:n_draw
        idx += 1
        cond = Turing.DynamicPPL.condition(
            mdl,
            (
                rw_init = chn[:rw_init][d, c],
                std = chn[:std][d, c],
                Symbol("ϵ_t") => chn[Symbol("ϵ_t")][d, c],
                init_incidence = chn[:init_incidence][d, c],
                Symbol("Ascertainment.intercept") =>
                    chn[Symbol("Ascertainment.intercept")][d, c],
                Symbol("σ") => chn[Symbol("σ")][d, c],
            ),
        )
        out = cond()
        I_draws[idx, :] .= out.I_t
        # generated_y_t may keep `missing` at unobserved/forecast positions.
        pred_draws[idx, :] .= coalesce.(out.generated_y_t, NaN)
        # Expected load pre-delay: Ascertainment's default transform is
        # xexpy (Y_t .* exp(x)), so load_t = I_t .* exp(lpc).
        load_draws[idx, :] .= out.I_t .* exp(lpc[idx])
    end
    return (; I_draws, pred_draws, load_draws)
end

function _median_ci(m)
    # Drop NaN rows (forecast blanks / missing at unobserved days) per column
    # before summarising.
    n_pts = size(m, 2)
    med = Vector{Float64}(undef, n_pts)
    lo = Vector{Float64}(undef, n_pts)
    hi = Vector{Float64}(undef, n_pts)
    for i in 1:n_pts
        col = m[:, i]
        col = col[.!isnan.(col)]
        if isempty(col)
            med[i] = lo[i] = hi[i] = NaN
        else
            med[i] = median(col)
            lo[i] = quantile(col, 0.025)
            hi[i] = quantile(col, 0.975)
        end
    end
    return med, lo, hi
end

function _series_plot(title, ylabel, med, lo, hi)
    t = 1:length(med)
    fig = Figure(size = (900, 420))
    ax = Axis(fig[1, 1]; title = title, xlabel = "day", ylabel = ylabel)
    band!(ax, t, lo, hi; color = (:steelblue, 0.3), label = "95% CI")
    lines!(ax, t, med; color = :steelblue, label = "median")
    axislegend(ax; position = :lt)
    return fig
end

# Data + model needed for the conditional evaluation.
_expr_data = EpiSewer.example_data()
_expr_y = _expr_data.measurements.concentration[WINDOW]
_expr_flow = Vector{Float64}(_expr_data.flows.flow[WINDOW])
_expr_mdl = as_turing_model(EpiSewer.model(), (y = _expr_y, flow = _expr_flow), length(_expr_y))

_gen = _extract_generated(chn, _expr_mdl)
if !isnothing(_gen)
    # --- Infections over time ---
    I_med, I_lo, I_hi = _median_ci(_gen.I_draws)
    save(
        joinpath(outdir, "ww_plot_infections.png"),
        _series_plot(
            "Estimated infections per day", "infections", I_med, I_lo, I_hi,
        ),
    )
    @info "Saved infections plot" path = "docs/fits/ww_plot_infections.png"

    # --- Expected load over time ---
    L_med, L_lo, L_hi = _median_ci(_gen.load_draws)
    save(
        joinpath(outdir, "ww_plot_load.png"),
        _series_plot(
            "Expected pathogen load per day", "expected load", L_med, L_lo, L_hi,
        ),
    )
    @info "Saved load plot" path = "docs/fits/ww_plot_load.png"

    # --- Concentration fit: observed vs posterior predictive ---
    p_med, p_lo, p_hi = _median_ci(_gen.pred_draws)
    t = 1:length(p_med)
    fig = Figure(size = (900, 420))
    ax = Axis(
        fig[1, 1]; title = "Wastewater concentration fit",
        xlabel = "day", ylabel = "concentration (gc/mL)",
    )
    band!(ax, t, p_lo, p_hi; color = (:steelblue, 0.3), label = "95% CI")
    lines!(ax, t, p_med; color = :steelblue, label = "median")
    # Observed concentrations (skipping missing) as points.
    obs_t = findall(!ismissing, _expr_y)
    scatter!(ax, obs_t, collect(skipmissing(_expr_y)); color = :black, markersize = 4, label = "observed")
    axislegend(ax; position = :rt)
    save(joinpath(outdir, "ww_plot_concentration.png"), fig)
    @info "Saved concentration plot" path = "docs/fits/ww_plot_concentration.png"
else
    @warn "Generated-quantity extraction skipped: required chain parameters missing."
end

# --- Prior vs posterior PairPlots for key scalar parameters -------------------
# Key scalar params: the load-per-case, observation noise, RW step std.
pnames = [string(k) for k in keys(chn)]
_missing_keys(["Parameter(std)", "Parameter(σ)", "Parameter(rw_init)"], pnames)
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
    data = EpiSewer.example_data()
    y = data.measurements.concentration[WINDOW]
    flow = Vector{Float64}(data.flows.flow[WINDOW])
    mdl = as_turing_model(EpiSewer.model(), (y = y, flow = flow), length(y))
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
