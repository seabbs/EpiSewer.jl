# EpiSewer.jl <img src="docs/src/assets/logo.svg" width="150" alt="episewer logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://epiaware.org/EpiSewer.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://epiaware.org/EpiSewer.jl/dev/) | [![Test](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg)](https://codecov.io/gh/seabbs/EpiSewer.jl) [![AD](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml) | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FEpiSewer&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FEpiSewer&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) |

| ForwardDiff | ReverseDiff (tape) | ReverseDiff (compiled) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff) | [![cov ReverseDiff compiled](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff-compiled)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff-compiled) | [![cov Enzyme forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

EpiSewer.jl provides a Bayesian generative model to estimate the effective reproduction number R_t and other transmission indicators from pathogen concentrations measured in wastewater. This is a Julia implementation using composable components from the EpiAware.org ecosystem.

## Why EpiSewer.jl?

- **Composable model**: the wastewater model is assembled from interchangeable infection and observation parts, each of which is a model in its own right, rather than one monolithic model.
- **Swap a part to change an assumption**: change the `R_t` process, a delay distribution, or the measurement noise on its own, and read the consequences back off the assembly.
- **One interface**: the assembly becomes a single [Turing](https://turinglang.org) model through `as_turing_model`, so the full Turing toolbox applies.
- **A port of the R package**: the default assembly is the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package's default model, on its example data.
- **Built from the ecosystem**: five structs cover a model whose R original is assembled from about twenty components; the rest is composition of existing `ComposableTuringIDModels.jl` pieces.

Posterior fitting is not yet demonstrated on the thinned example data. The "Fitting it" subsection below records what has been measured.

### The model in words

This describes the model the EpiSewer R package implements.
EpiSewer.jl replicates part of it.
The [Model components](https://epiaware.org/EpiSewer.jl/stable/components/model-components) page maps each piece onto the ecosystem component that provides it, and sets out which parts the default Julia chain composes and which sit outside it.

**Data.** The model is driven by a time series of wastewater concentration measurements, ideally in gene copies per millilitre. Not every day needs a measurement: the model accounts for missing or non-daily observations. Alongside the measurements, daily wastewater flow records normalise the concentration signal for day-to-day variation in flow, for example due to rainfall. Confirmed case counts from the catchment area are optional, and calibrate the model so that the estimated number of infections roughly matches observed cases.

**From data to signal.** Raw concentrations are normalised by the daily flow, which removes much of the noise a varying flow introduces. The normalised signal is a delayed and blurred trace of the infection process upstream: infected individuals begin to shed the pathogen some time after infection and continue to shed over a period of days, and the shed load is blurred and delayed further as it travels through the sewer network to the sampling site.

**Assumptions.** Turning concentrations back into transmission dynamics needs a set of disease-specific distributions, typically taken from the literature: the generation time, the shedding load distribution, and, because the shedding load is usually expressed relative to symptom onset, the incubation period. All are discretised to a daily time step.

**The latent model.** Underlying the observations is a latent epidemic process. New infections each day evolve according to a renewal process driven by a time-varying reproduction number, `R_t`, which itself varies smoothly over time under a flexible smoothing model such as a Gaussian process or a random walk. The R model's infection process also carries stochastic noise beyond what a deterministic renewal implies.

**The observation model.** Infections are mapped to an expected load by convolving with the shedding load distribution. In the R model that load is then delayed through the sewer residence time before being divided by the flow to give an expected concentration. Observed concentrations are that expected concentration corrupted by measurement noise.

**Estimation.** Together these give a full Bayesian generative model, fitted with Hamiltonian MCMC to produce a posterior over every parameter, and from it `R_t` and the other transmission indicators with credible intervals.

### Derived from EpiSewer

This package is a Julia port of the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package by Adrian Lison and colleagues. The original model is described in:

> Lison, A., McLeod, R.E., Huisman, J.S. et al. Real-Time Estimation of Pathogen Transmission Dynamics from Wastewater. *Nature Communications* (2026). DOI: [10.1038/s41467-026-75380-3](https://doi.org/10.1038/s41467-026-75380-3)

This Julia version uses composable model components from the EpiAware.org ecosystem (`ComposableTuringIDModels.jl` and `EpiAwareADTools.jl`) in place of the Stan implementation of the original.
The delay distributions are discretised by double-interval censoring, as in the original.
That discretisation happens inside the components: `Renewal` and `LatentDelay` call `double_interval_censored` themselves, so it reaches this package through `ComposableTuringIDModels.jl` rather than through a direct dependency on `CensoredDistributions.jl`.

## Getting started

You assemble the wastewater model from interchangeable parts, and EpiSewer.jl turns the assembly into a single Turing model you can draw from and fit.
`EpiSewer.model` is **public but not exported**, so call it as `EpiSewer.model(...)`, never `model(...)`.

Load the Zurich SARS-CoV-2 example data and thin the measurements to Mondays and Thursdays.
This is the artificially sparse series the EpiSewer README example fits.
The withheld days are left for the model to fill in.

```julia
using EpiSewer, ComposableTuringIDModels, Turing
using Dates: dayname
import ComposableTuringIDModels: as_turing_model

d = EpiSewer.example_data()
flow = Vector{Float64}(d.flows.flow)      # daily flow (mL/day), data
sparse_days = dayname.(d.measurements.date) .∈ (["Monday", "Thursday"],)
y = ifelse.(sparse_days, d.measurements.concentration, missing)
(days = length(y), observed = count(!ismissing, y))
```

Compose the model.
The defaults are the R package's default model and the assumptions its README uses: a shifted-Gamma generation time (mean 3, sd 2.4), a `Gamma(0.929639, 7.241397)` shedding load, and a `Gamma(8.5, 0.4)` incubation period.

```julia
mdl = EpiSewer.model()
```

That prints as a tree of named parts rather than one monolithic model, which is the point of the port.
The parts are a `Renewal` infection process with a random-walk `R_t`, observed through the incubation delay, the per-case shed load, the shedding delay, division by flow, and log-normal measurement noise.

Size the infection series.
Each `LatentDelay` in the observation chain drops the partially observed head of its convolution, so the infections need the chain's lead-in on top of the observed days.
Passing `n = length(y)` instead would leave the leading observations unscored.

```julia
n = length(y) + EpiSewer.observation_lead_in(mdl)
```

Build the Turing model.
The concentrations and the daily flow are both data.
They travel together through the observation-data contract rather than as arguments to `EpiSewer.model`.

```julia
mdl_t = as_turing_model(mdl, (y = y, flow = flow), n)
```

Draw from the prior and inspect the generated quantities.

```julia
prior = sample(mdl_t, Prior(), 200; progress = false)
draws = vec(returned(mdl_t, prior))
first_draw = first(draws)
map(x -> round.(extrema(x); sigdigits = 3),
    (R_t = exp.(first_draw.Z_t), I_t = first_draw.I_t,
        concentration = first_draw.generated_y_t))
```

Summarise `R_t` across the prior draws.
`Renewal` samples a latent `Z_t` and applies its `exp` transformation, so `R_t = exp.(Z_t)`.

```julia
using Statistics: median, quantile

Rt_draws = reduce(hcat, [exp.(g.Z_t) for g in draws])
Rt_median = [median(view(Rt_draws, i, :)) for i in axes(Rt_draws, 1)]
(median = round(median(Rt_median); digits = 2),
    range = round.(extrema(Rt_median); digits = 2))
```

### Figures

The docs build executes these blocks, so the figures on the rendered documentation page are produced by the example's own code.
Nothing is committed as an image.

```julia
using CairoMakie, Dates

# `band!` takes numeric x, so dates are plotted as day numbers and the ticks
# carry the formatted dates.
xnum(dates) = Dates.value.(dates)
function month_ticks(dates)
    ms = filter(dt -> Dates.day(dt) == 1, dates)
    return (xnum(ms), Dates.format.(ms, "u yyyy"))
end
```

The measurements the model is given, against the ones the thinning withholds.
The flow that normalises them is below.

```julia
obs_dates = d.measurements.date
kept = .!ismissing.(y)
withheld = ismissing.(y) .& .!ismissing.(d.measurements.concentration)

fig = Figure(size = (820, 460))
ax = Axis(fig[1, 1]; ylabel = "gc/mL", yscale = log10,
    xticks = month_ticks(obs_dates),
    title = "Measurements, thinned to Mondays and Thursdays")
scatter!(ax, xnum(obs_dates[withheld]),
    Float64.(d.measurements.concentration[withheld]);
    color = (:grey, 0.55), marker = :xcross, markersize = 7, label = "withheld")
scatter!(ax, xnum(obs_dates[kept]), Float64.(y[kept]);
    color = :black, markersize = 7, label = "given to the model")
axislegend(ax; position = :rt, framevisible = false)
ax_flow = Axis(fig[2, 1]; ylabel = "flow (mL/day)",
    xticks = month_ticks(obs_dates))
lines!(ax_flow, xnum(obs_dates), flow; color = :seagreen)
fig
```

The prior on `R_t`, over the infection series including its lead-in.

```julia
inf_dates = collect(
    (first(obs_dates) - Day(EpiSewer.observation_lead_in(mdl))):Day(1):last(obs_dates))
band_q(m, p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
x = xnum(inf_dates)

fig = Figure(size = (820, 290))
ax = Axis(fig[1, 1]; ylabel = "R_t", yscale = log10,
    xticks = month_ticks(inf_dates),
    title = "Prior R_t: median, 50% and 95% intervals")
band!(ax, x, band_q(Rt_draws, 0.025), band_q(Rt_draws, 0.975);
    color = (:steelblue, 0.15))
band!(ax, x, band_q(Rt_draws, 0.25), band_q(Rt_draws, 0.75);
    color = (:steelblue, 0.35))
lines!(ax, x, band_q(Rt_draws, 0.5); color = :steelblue, linewidth = 2)
hlines!(ax, [1.0]; color = :grey, linestyle = :dash)
fig
```

The R package's README also plots the concentration fit, the expected load and the infections.
Those need a posterior.
Drawn from the prior they carry no information: on a withheld day the 95% prior predictive concentration runs from 9e-7 to 7e57 gc/mL, about 64 orders of magnitude, because a 164-step random walk on `log R_t` compounds through the renewal process.
The measurements themselves span 145 to 3200 gc/mL.
`R_t` is the one prior quantity with a readable scale, so it is the one drawn.

Note also that `generated_y_t` returns the observed value on a day that has one, and a draw only where the series is `missing`.
A ribbon built from it is therefore part data and part prior, which is why the measurements are plotted on their own above.

### Fitting it

Everything above runs in seconds.
Posterior inference does not, and it is not yet demonstrated.
Three things have been measured and are worth keeping apart.

**Wall clock.** The model has 255 parameters: 164 random-walk steps, 87 withheld days that the model imputes, and 4 scalars.
NUTS adapts to a step size near 0.001 and then sits at maximum tree depth, so one iteration costs hundreds of gradients.
The gradient is the part that is solved.
Mooncake costs 1.4 ms against ForwardDiff's 24 ms on this model, and Turing's default is ForwardDiff, so the backend has to be set explicitly.
Even so, 2 chains of 300 warmup and 300 draws took 19 minutes.

**Convergence.** That run returned a maximum R-hat of 2.8, a minimum effective sample size of 2, and 122 of its 255 parameters above 1.05.
4 chains of 400 warmup and 400 draws did not finish inside 30 minutes.

**Data density.** At identical settings the same model samples with no divergent transitions on the dense series of 117 measurements, and produces 53 in a 100-iteration run on the thinned 33.
That points at weak identification rather than a fault in the port: 33 measurements do not pin down 164 random-walk steps.

Why the R package fits this same thinned series is not established here.
Its Gaussian-process prior on `R_t` may regularise where a random walk does not, or Stan's parameterisation may handle the geometry better.
Tracked in [#16](https://github.com/seabbs/EpiSewer.jl/issues/16).

The settings a fit needs:

```jl
import Mooncake

chn = sample(
    mdl_t, NUTS(0.9; adtype = Turing.AutoMooncake()),
    MCMCThreads(), 400, 4; warmup = 400, progress = false)

Rt_post = reduce(hcat, [exp.(g.Z_t) for g in vec(returned(mdl_t, chn))])
```

That block is fenced ` ```jl ` rather than ` ```julia ` on purpose.
The docs build executes every ` ```julia ` fence in this README, and a fit of this length does not belong in a build that runs on every push.

Check `MCMCChains.gelmandiag` and the divergence count before reading anything off a chain.
Treat `R_t` near the end of the series with care either way.
As the R package's README also notes, wastewater arrives delayed, so the most recent days are informed by very little data.

## Swap a component

Every stage is interchangeable, so a changed modelling assumption is a changed argument rather than a new model.
`infection_model` and `observation_model` replace a whole stage.
The delay keywords replace one distribution.

Same observation chain, a different `R_t` process.
Here an AR(1) rather than a random walk.

```julia
using Distributions: Gamma, Normal

gen = Gamma(((3.0 - 1) / 2.4)^2, 2.4^2 / (3.0 - 1)) + 1
ar_mdl = EpiSewer.model(
    infection_model = Renewal(;
        generation_time = gen, rt = AR(), initialisation = Normal(),
        D_gen = 15.0))
```

Adding a sewer residence time lengthens the observation chain, so read `n` back from `observation_lead_in` rather than hard-coding it.

```julia
res_mdl = EpiSewer.model(residence_dist = Gamma(2.0, 1.0), D_residence = 5.0)
(default = EpiSewer.observation_lead_in(mdl),
    with_residence = EpiSewer.observation_lead_in(res_mdl))
```

## Related packages

- [ComposableTuringIDModels.jl](https://github.com/EpiAware/ComposableTuringIDModels.jl): composable ID models built on Turing.
- [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl): discretised and censored distributions for delay processes.
- [EpiAwareADTools.jl](https://github.com/EpiAware/EpiAwareADTools.jl): automatic-differentiation tooling for the EpiAware ecosystem.

## Documentation

Full documentation is hosted at [epiaware.org/EpiSewer.jl](https://epiaware.org/EpiSewer.jl/stable/), where the home page is this README with its example executed and its figures rendered.

- **Model components** maps each component of the R model onto the ecosystem piece that provides it, and marks the boundary of the default chain.
- **Public API** documents the exported and public interface.
- **LLM-assisted development process** records how this port was produced and reviewed.

<!-- standard-sections:start -->
<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit between the
     markers. These standard sections are re-rendered on every update;
     edit the package-owned sections outside them, or CITATION.cff. -->

## Contributing

We welcome contributions and new contributors! Please open an issue or pull request on [GitHub](https://github.com/seabbs/EpiSewer.jl). This package follows [ColPrac](https://github.com/SciML/ColPrac) and is formatted with [Runic](https://github.com/fredrikekre/Runic.jl).

## How to cite

If you use EpiSewer in your work, please cite it. Citation metadata lives in [`CITATION.cff`](https://github.com/seabbs/EpiSewer.jl/blob/main/CITATION.cff), which GitHub renders as a "Cite this repository" button on the repository page.

## Code of conduct

Please note that the EpiSewer project is released with a [Contributor Code of Conduct](https://github.com/EpiAware/.github/blob/main/CODE_OF_CONDUCT.md). By contributing, you agree to abide by its terms.
<!-- standard-sections:end -->
