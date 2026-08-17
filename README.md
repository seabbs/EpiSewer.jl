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

- **Composable model**: the wastewater model is assembled from interchangeable infection and observation parts, each of which is a model in its own right.
- **The model**: infections follow a renewal process driven by a smoothly varying `R_t`; convolving them with an incubation period and a shedding load profile gives the pathogen load arriving at the sampling site, which divided by the daily flow gives the measured concentration.
- **Swap a part to change an assumption**: change the `R_t` process, a delay distribution, or the measurement noise on its own, and read the consequences back off the assembly.
- **One interface**: the assembly becomes a single [Turing](https://turinglang.org) model through `as_turing_model`, so the full Turing toolbox applies.
- **The R package's defaults**: the default assembly is the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package's default model, with its priors, on its example data. Missing and non-daily measurements are handled as they are there.
- **Built from the ecosystem**: five structs cover a model whose R original is assembled from about twenty components, and the rest is composition of existing `ComposableTuringIDModels.jl` pieces.

The [Model components](https://epiaware.org/EpiSewer.jl/stable/components/model-components) page maps each component of the R model onto the ecosystem piece that provides it, and marks the boundary of the default chain.

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
The infection process is a `Renewal` with stochastic infections, over an `R_t` prior built from a short-term and a long-term `HilbertSpaceGP` summed under a softplus link.
It is observed through the incubation delay, the per-case shed load, the shedding delay, division by flow, and log-normal measurement noise.

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
prior = sample(mdl_t, Prior(), 300; progress = false)
draws = vec(returned(mdl_t, prior))
first_draw = first(draws)
map(x -> round.(extrema(x); sigdigits = 3),
    (R_t = exp.(first_draw.Z_t), I_t = first_draw.I_t,
        concentration = first_draw.expected_y_t))
```

### Figures

The docs build executes these blocks, so the figures on the rendered documentation page are produced by this code.

```julia
using AlgebraOfGraphics, CairoMakie, DataFrames
using Statistics: quantile
using Dates: Day

# Quantile bands across draws, one row per time point.
function bands(m, dates)
    q(p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
    return DataFrame(date = dates, med = q(0.5), lo = q(0.25), hi = q(0.75),
        lo95 = q(0.025), hi95 = q(0.975))
end

# A median line over its 50% interval.
ribbon(df) = data(df) * (
    mapping(:date, :lo, :hi) * visual(Band; alpha = 0.35, color = :steelblue) +
        mapping(:date, :med) * visual(Lines; linewidth = 2, color = :steelblue))

# The 95% interval underneath, for a quantity whose tails fit on the axis.
outer(df) = data(df) * mapping(:date, :lo95, :hi95) *
    visual(Band; alpha = 0.18, color = :steelblue)

# One generated quantity across the draws, as a time-by-draw matrix.
across(f) = reduce(hcat, [f(g) for g in draws])

obs_dates = d.measurements.date
inf_dates = collect(
    (first(obs_dates) - Day(EpiSewer.observation_lead_in(mdl))):Day(1):last(obs_dates))
```

The concentration the model expects, against the measurements it was given and the ones the thinning withheld.

```julia
points = dropmissing(DataFrame(date = obs_dates,
    concentration = d.measurements.concentration,
    measurement = ifelse.(.!ismissing.(y), "given to the model", "withheld")))

draw(
    ribbon(bands(across(g -> g.expected_y_t), obs_dates)) +
        data(points) * mapping(:date, :concentration,
        color = :measurement, marker = :measurement) *
        visual(Scatter; markersize = 8),
    scales(Color = (; palette = [:black, :grey55]),
        Marker = (; palette = [:circle, :xcross]));
    axis = (; yscale = log10, ylabel = "gc/mL",
        title = "Prior predictive concentration: median and 50% interval"),
    figure = (; size = (860, 330))
)
```

`R_t`, over the infection series including its lead-in.
`Renewal` samples a latent `Z_t` and applies its `exp` transformation, so `R_t = exp.(Z_t)`.

```julia
rt = bands(across(g -> exp.(g.Z_t)), inf_dates)

draw(outer(rt) + ribbon(rt);
    axis = (; ylabel = "R_t",
        title = "Prior R_t: median, 50% and 95% intervals"),
    figure = (; size = (860, 300)))
```

The two latent series behind a measurement are the load arriving at the sampling site each day and the infections that shed it.
Both carry the 50% interval only, because their 95% interval spans about eighteen orders of magnitude and no axis carries that usefully.
Even the 50% runs to six, so the axis is held to the median trajectory and the interval is clipped where it leaves the panel.
What that shows is the slow upward drift the long-term Gaussian process puts in the prior.

```julia
# Hold the axis to the median trajectory. The interval runs wider than any
# axis can show, so it is clipped rather than allowed to set the range.
function median_limits(b; decades = 1.5)
    lo, hi = extrema(b.med)
    return (nothing, (lo / 10^decades, hi * 10^decades))
end

load = bands(across(g -> g.expected_y_t .* flow), obs_dates)

draw(ribbon(load);
    axis = (; yscale = log10, ylabel = "gc/day",
        limits = median_limits(load),
        title = "Prior expected load: median and 50% interval"),
    figure = (; size = (860, 290)))
```

```julia
infections = bands(across(g -> g.I_t), inf_dates)

draw(ribbon(infections);
    axis = (; yscale = log10, ylabel = "infections",
        limits = median_limits(infections),
        title = "Prior infections: median and 50% interval"),
    figure = (; size = (860, 290)))
```

Fit with NUTS.
Turing defaults to `ForwardDiff`; set a reverse-mode backend, which is much cheaper per gradient at this size.

```jl
import Mooncake

chn = sample(mdl_t, NUTS(0.9; adtype = Turing.AutoMooncake()),
    MCMCThreads(), 500, 4; warmup = 500)
```

Substituting `vec(returned(mdl_t, chn))` for `draws` turns every figure above into a posterior summary.

That block is fenced ` ```jl ` rather than ` ```julia ` on purpose.
The docs build executes every ` ```julia ` fence in this README, and sampling belongs outside a build that runs on every push.

### Swapping a component

Every stage is interchangeable, so a changed modelling assumption is a changed argument rather than a new model.
`rt` replaces the `R_t` prior, `infection_model` and `observation_model` replace a whole stage, and the delay keywords replace one distribution.

The default `R_t` prior is the R package's pair of approximate Gaussian processes.
A random walk instead:

```julia
rw_mdl = EpiSewer.model(rt = RandomWalk())
rw_mdl.infection_model.rt
```

Adding a sewer residence time lengthens the observation chain, so read `n` back from `observation_lead_in` rather than hard-coding it.

```julia
using Distributions: Gamma

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
