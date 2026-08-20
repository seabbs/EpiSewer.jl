# EpiSewer.jl <img src="docs/src/assets/logo.svg" width="150" alt="episewer logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://samabbott.co.uk/EpiSewer.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://samabbott.co.uk/EpiSewer.jl/dev/) | [![Test](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg)](https://codecov.io/gh/seabbs/EpiSewer.jl) [![AD](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml) | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FEpiSewer&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FEpiSewer&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) |

| ForwardDiff | ReverseDiff (tape) | ReverseDiff (compiled) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff) | [![cov ReverseDiff compiled](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff-compiled)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff-compiled) | [![cov Enzyme forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

*A Julia port of the [EpiSewer](https://github.com/adrian-lison/EpiSewer)
wastewater model, built from composable EpiAware components.*

## Why EpiSewer.jl?

- Wastewater measures transmission without depending on who comes forward for
  a test, so it keeps reporting when testing falls away.
- Every assumption is an argument, so changing the `R_t` process or the
  measurement noise means passing a different component rather than editing
  model code.
- The assembly becomes a [Turing](https://turinglang.org) model, so the
  samplers, diagnostics and plotting you already use apply to it unchanged.
- Concentrations and case counts can be fitted as one model, instead of two
  pipelines whose estimates are combined afterwards.
- Missing days, non-daily sampling and non-detects are part of the likelihood,
  so a sparse series needs no imputation before it is fitted.

### Derived from EpiSewer

This package is a Julia port of the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package by Adrian Lison and colleagues. The original model is described in:

> Lison, A., McLeod, R.E., Huisman, J.S. et al. Real-Time Estimation of Pathogen Transmission Dynamics from Wastewater. *Nature Communications* (2026). DOI: [10.1038/s41467-026-75380-3](https://doi.org/10.1038/s41467-026-75380-3)

This Julia version uses composable model components from the EpiAware.org ecosystem (`ComposableTuringIDModels.jl` and `EpiAwareADTools.jl`) in place of the Stan implementation of the original.
Delay distributions are discretised by double-interval censoring, as in the original.

## Getting started

### Data

EpiSewer.jl needs a time series of pathogen concentration measurements and the daily wastewater volume flowing through the sampling site.
The flow matters because a concentration is a shed load diluted by whatever water passed the site that day.
The example data are daily SARS-CoV-2 (N1 gene) concentrations in gc/mL and flows in mL/day at the Zurich treatment plant, provided by EAWAG to the public domain.

To show the handling of missing data, we make the series artificially sparse by keeping only the measurements taken on Mondays and Thursdays.
The withheld days are left for the model to fill in.

```julia
using EpiSewer, ComposableTuringIDModels, DataFrames, DataFramesMeta, Turing
using Dates: dayname

d = EpiSewer.example_data()
measurements = @chain d.measurements begin
    @rtransform :sparse = ifelse(
        dayname(:date) ∈ ("Monday", "Thursday"), :concentration, missing
    )
end
y = measurements.sparse
flow = Vector{Float64}(d.flows.flow)
(days = length(y), measured = count(!ismissing, y))
```

### Assumptions

Estimating transmission from concentrations needs assumptions the data cannot supply.

- the generation time
- the load shed over time by an average infected individual
- the incubation period, which puts that profile on the infection timescale
- the load shed per case, which sets the scale of the whole series
- the scale of infections at the start of the series

The first three travel with the pathogen and are usually taken from the literature, while the last two are properties of this site and series.
The `EpiSewer.model` entry point asks for all five rather than assuming them, as EpiSewer's own `sewer_assumptions` asks for the first four.
For this example, `example_assumptions` holds the set the EpiSewer README uses for SARS-CoV-2 in Zurich.

```julia
EpiSewer.example_assumptions()
```

The delays are continuous distributions here, and the components that use them discretise them by double-interval censoring.

### Estimation

`EpiSewer.model` assembles the default model from those assumptions.
Infections follow a renewal process with a smoothly varying `R_t`, and an observation chain turns them into a concentration.
That chain spreads each infection's load over the incubation period and the shedding profile, varies the load shed between individuals, scales by the load shed per case, and divides by the daily flow.
Outlier spikes and relative log-normal noise then account for what the measurement itself adds.

```julia
idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
```

The infection series starts before the first measurement, because each delay in the observation chain consumes part of it.
`observation_lead_in` gives that lead-in, and the concentrations and flows travel together as data.

```julia
n = length(y) + EpiSewer.observation_lead_in(idm)
mdl = as_turing_model(idm, (y = y, flow = flow), n)
```

Fitting uses Turing's `NUTS` with reverse-mode gradients from `Mooncake`, over two chains.

```julia
import Mooncake
using MCMCChains: summarystats

chn = sample(
    mdl, NUTS(0.9; adtype = Turing.AutoMooncake(), max_depth = 8),
    MCMCThreads(), 250, 2; num_warmup = 250, progress = false
)
draws = vec(returned(mdl, chn))
summarystats(chn[[@varname(cv), @varname(init_incidence)]])
```

The summary covers two of the fitted parameters: `cv`, the coefficient of variation of the measurement noise, and `init_incidence`, the log of the infections seeding the series.
The latent series are computed rather than sampled, so `returned` replays the model over the posterior samples to recover them.
Each draw then carries `R_t` on the log scale as `Z_t`, infections as `I_t`, and the expected concentration as `expected_y_t`.

### The posterior predictive

The figures show the posterior predictive concentration rather than the expected concentration, which is what EpiSewer plots, since its `plot_concentration` defaults to `include_noise = TRUE`.
The expected concentration alone omits the measurement error, so it gives a narrower band than the data it is drawn against.

`observation_error` gives the error distribution at an expected value and a coefficient of variation, so one draw from it per posterior draw is a predictive replicate.

```julia
using Distributions: rand

cv_draws = vec(chn[:cv])
predicted = [
    rand.(observation_error.(Ref(EpiSewer.LogNormalError()), g.expected_y_t, c))
        for (g, c) in zip(draws, cv_draws)
]
```

### Plotting the results

The figures below share two helpers: quantiles of a generated quantity across the draws, and a median line over its 50% and 95% credible intervals.

```julia
using AlgebraOfGraphics, CairoMakie
using Statistics: quantile
using Dates: Day

function summarise(draws, f, dates)
    m = reduce(hcat, [f(g) for g in draws])
    q(p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
    return DataFrame(
        date = dates, med = q(0.5), lo50 = q(0.25), hi50 = q(0.75),
        lo95 = q(0.025), hi95 = q(0.975)
    )
end

ribbon(df) = data(df) * (
    mapping(:date, :lo95, :hi95) *
        visual(Band; alpha = 0.2, color = :steelblue) +
        mapping(:date, :lo50, :hi50) *
        visual(Band; alpha = 0.4, color = :steelblue) +
        mapping(:date, :med) * visual(Lines; linewidth = 2, color = :steelblue)
)

function series_plot(df; reference = nothing, kwargs...)
    layers = ribbon(df)
    if reference !== nothing
        layers += data((; y = [reference])) * mapping(:y) *
            visual(HLines; color = :grey40, linestyle = :dash)
    end
    return draw(layers; axis = (; kwargs...), figure = (; size = (860, 300)))
end

obs_dates = measurements.date
inf_dates = (first(obs_dates) - Day(n - length(y))):Day(1):last(obs_dates)
```

#### Model fit

It is good practice to first check how well the model fitted the data, by plotting the observed measurements against the ones the model expects.
The black dots are the measurements the model was given, and the grey crosses the ones the thinning withheld.

```julia
observed = @chain measurements begin
    @rsubset !ismissing(:concentration)
    @rtransform :measurement = ismissing(:sparse) ? "withheld" : "given"
end

function concentration_plot(df, points)
    layers = ribbon(df) +
        data(points) * mapping(
        :date, :concentration, color = :measurement, marker = :measurement
    ) * visual(Scatter; markersize = 8)
    return draw(
        layers,
        scales(
            Color = (; palette = [:black, :grey55]),
            Marker = (; palette = [:circle, :xcross])
        );
        axis = (; yscale = log10, ylabel = "concentration (gc/mL)"),
        figure = (; size = (860, 340))
    )
end

concentration_plot(summarise(predicted, identity, obs_dates), observed)
```

#### Time-varying effective reproduction number

`Renewal` samples `R_t` on the log scale as `Z_t`, so the estimates are `exp.(Z_t)`.

```julia
series_plot(
    summarise(draws, g -> exp.(g.Z_t), inf_dates);
    reference = 1.0, ylabel = "R_t"
)
```

The estimates reach further back than the measurements, because a concentration measured today is mostly a signal of infections some days earlier.
The estimates closest to the present are the most uncertain for the same reason, since only part of the transmission they describe has reached the sampling site.

#### Growth rate

The growth rate follows from `R_t` and the generation time.

```julia
series_plot(
    summarise(
        draws, g -> R_to_r.(exp.(g.Z_t), Ref(idm.infection_model)), inf_dates
    );
    reference = 0.0, ylabel = "growth rate (per day)"
)
```

#### Latent parameters

The other series in the model are worth inspecting as a check on the fit.
The expected load is the total load arriving at the sampling site on a given day, before it is diluted by the flow.

```julia
series_plot(
    summarise(draws, g -> g.expected_y_t .* flow, obs_dates);
    ylabel = "load (gc/day)"
)
```

Infections follow a similar trend, ahead of the load and less smooth, because an infected individual starts shedding only after their infection and then sheds over several weeks.
These are relative rather than absolute infections.
Their scale depends on the assumed load shed per case, so they should not be read as incidence.

```julia
series_plot(summarise(draws, g -> g.I_t, inf_dates); ylabel = "infections")
```

## Related packages

- [ComposableTuringIDModels.jl](https://github.com/EpiAware/ComposableTuringIDModels.jl): composable ID models built on Turing.
- [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl): discretised and censored distributions for delay processes.
- [EpiAwareADTools.jl](https://github.com/EpiAware/EpiAwareADTools.jl): automatic-differentiation tooling for the EpiAware ecosystem.

## Documentation

Full documentation is hosted at [samabbott.co.uk/EpiSewer.jl](https://samabbott.co.uk/EpiSewer.jl/stable/).

- **Getting started** covers the entry point, the two places a whole stage can be swapped, and what a swap does to the model.
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
