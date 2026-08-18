# EpiSewer.jl <img src="docs/src/assets/logo.svg" width="150" alt="episewer logo" align="right">

<!-- badges:start -->
| **Documentation** | **Build Status** | **Code Quality** | **License & DOI** | **Downloads** |
|:-----------------:|:----------------:|:----------------:|:-----------------:|:-------------:|
| [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://samabbott.co.uk/EpiSewer.jl/stable/) [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://samabbott.co.uk/EpiSewer.jl/dev/) | [![Test](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg)](https://codecov.io/gh/seabbs/EpiSewer.jl) [![AD](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml/badge.svg?branch=main)](https://github.com/seabbs/EpiSewer.jl/actions/workflows/ad.yaml) | [![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl) [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) [![JET](https://img.shields.io/badge/%E2%9C%88%EF%B8%8F%20tested%20with%20-%20JET.jl%20-%20red)](https://github.com/aviatesk/JET.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) | [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FEpiSewer&query=total_requests&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) [![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FEpiSewer&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/EpiSewer) |

| ForwardDiff | ReverseDiff (tape) | ReverseDiff (compiled) | Enzyme forward | Enzyme reverse | Mooncake reverse | Mooncake forward |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [![cov ForwardDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-forwarddiff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-forwarddiff) | [![cov ReverseDiff](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff) | [![cov ReverseDiff compiled](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-reversediff-compiled)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-reversediff-compiled) | [![cov Enzyme forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-forward) | [![cov Enzyme reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-enzyme-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-enzyme-reverse) | [![cov Mooncake reverse](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-reverse)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-reverse) | [![cov Mooncake forward](https://codecov.io/gh/seabbs/EpiSewer.jl/graph/badge.svg?flag=ad-mooncake-forward)](https://app.codecov.io/gh/seabbs/EpiSewer.jl?flags%5B0%5D=ad-mooncake-forward) |
<!-- badges:end -->

*A Julia port of the EpiSewer wastewater model, built from composable
EpiAware components.*

## Why EpiSewer.jl?

- **Transmission from wastewater**: estimate `R_t`, infections and shed load
  from pathogen concentrations, with missing and non-daily measurements carried
  by the model rather than imputed beforehand.
- **Swap an assumption, not a model**: each stage is a separate component, so
  changing the `R_t` process, a delay distribution, or the measurement noise is
  a changed argument rather than a rewrite.
- **One interface**: the assembly is a `ComposableTuringIDModels.IDModel` and
  becomes a [Turing](https://turinglang.org) model through `as_turing_model`,
  so the full Turing toolbox applies and it nests with the rest of the
  ecosystem.
- **More than one data stream**: the same infection process feeds wastewater
  concentrations and case surveillance together through `Split`, each with its
  own delays and noise.
- **A library of parts**: relative log-normal noise, digital PCR counts, limits
  of detection, outlier spikes, flow normalisation, sewer residence time,
  shedding load profiles, load shed per case, individual-level load variation,
  and stochastic infections.

The [Model components](https://samabbott.co.uk/EpiSewer.jl/stable/components/model-components) page maps each component of the R model onto the ecosystem piece that provides it, and marks the boundary of the default chain.

### Derived from EpiSewer

This package is a Julia port of the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package by Adrian Lison and colleagues. The original model is described in:

> Lison, A., McLeod, R.E., Huisman, J.S. et al. Real-Time Estimation of Pathogen Transmission Dynamics from Wastewater. *Nature Communications* (2026). DOI: [10.1038/s41467-026-75380-3](https://doi.org/10.1038/s41467-026-75380-3)

This Julia version uses composable model components from the EpiAware.org ecosystem (`ComposableTuringIDModels.jl` and `EpiAwareADTools.jl`) in place of the Stan implementation of the original.
Delay distributions are discretised by double-interval censoring, as in the original.

## Getting started

### Data

EpiSewer.jl needs a time series of pathogen concentration measurements and the daily wastewater volume flowing through the sampling site.
The example data are daily SARS-CoV-2 (N1 gene) concentrations in gc/mL and flows in mL/day at the Zurich treatment plant, provided by EAWAG to the public domain.

To show the handling of missing data, we make the series artificially sparse by keeping only the measurements taken on Mondays and Thursdays.
The withheld days are left for the model to fill in.

```julia
using EpiSewer, DataFrames, DataFramesMeta, Turing
import ComposableTuringIDModels as CT
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

Estimating transmission from concentrations requires assumptions about the disease: the generation time, the load shed over time by an average infected individual, the incubation period that puts that profile on the infection timescale, and the scale of infections at the start of the series.
These are disease-specific and typically taken from the literature, so `EpiSewer.model` asks for them rather than assuming them.
`example_assumptions` holds the set the EpiSewer README uses for SARS-CoV-2 in Zurich.

```julia
EpiSewer.example_assumptions()
```

The delays are continuous distributions here, and the components that use them discretise them by double-interval censoring.

### Estimation

`EpiSewer.model` assembles the default model.
Infections follow a renewal process with a smoothly varying `R_t`.
They are observed through the incubation period, variation in the load shed between individuals, the load shed per case, the shedding profile, division by the daily flow, outlier spikes, and relative log-normal measurement noise.

```julia
idm = EpiSewer.model(; EpiSewer.example_assumptions()...)
```

The infection series starts before the first measurement, because each delay in the observation chain consumes part of it.
`observation_lead_in` gives that lead-in, and the concentrations and flows travel together as data.

```julia
n = length(y) + EpiSewer.observation_lead_in(idm)
mdl = CT.as_turing_model(idm, (y = y, flow = flow), n)
```

Fitting is Turing's `NUTS`, over reverse-mode gradients from `Mooncake`, on two chains.

```julia
import Mooncake

chn = sample(
    mdl, NUTS(0.9; adtype = Turing.AutoMooncake(), max_depth = 8),
    MCMCThreads(), 250, 2; num_warmup = 250, progress = false
)
draws = vec(returned(mdl, chn))
```

`returned` replays the model over the posterior samples, giving the latent series behind each draw: `R_t` on the log scale as `Z_t`, infections as `I_t`, and the expected concentration as `expected_y_t`.

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

concentration_plot(summarise(draws, g -> g.expected_y_t, obs_dates), observed)
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
For the same reason the estimates closest to the present are the most uncertain: only part of the transmission they describe has reached the sampling site.

#### Growth rate

The growth rate follows from `R_t` and the generation time.

```julia
series_plot(
    summarise(
        draws, g -> CT.R_to_r.(exp.(g.Z_t), Ref(idm.infection_model)), inf_dates
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
These are relative rather than absolute infections: their scale depends on the assumed load shed per case, so they should not be read as incidence.

```julia
series_plot(summarise(draws, g -> g.I_t, inf_dates); ylabel = "infections")
```

## Related packages

- [ComposableTuringIDModels.jl](https://github.com/EpiAware/ComposableTuringIDModels.jl): composable ID models built on Turing.
- [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl): discretised and censored distributions for delay processes.
- [EpiAwareADTools.jl](https://github.com/EpiAware/EpiAwareADTools.jl): automatic-differentiation tooling for the EpiAware ecosystem.

## Documentation

Full documentation is hosted at [samabbott.co.uk/EpiSewer.jl](https://samabbott.co.uk/EpiSewer.jl/stable/).

- **Getting started** covers the components this package adds and how to customise the model.
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
