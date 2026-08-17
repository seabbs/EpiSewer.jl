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

## About

The package replicates the modelling functionality of the EpiSewer R package by Adrian Lison. It estimates transmission dynamics from wastewater concentration measurements using MCMC sampling, with full uncertainty quantification.

## The model in words

This section walks through the model that EpiSewer.jl replicates, in plain language. The worked example under "Getting started" below runs it.

### Data

The model is driven by a time series of wastewater concentration measurements, ideally in gene copies per millilitre. Not every day needs a measurement: the model naturally accounts for missing or non-daily observations. Alongside the measurements, daily wastewater flow records are used to normalise the concentration signal for day-to-day variation in flow (for example due to rainfall). Optionally, confirmed case counts from the catchment area can be supplied; these calibrate the model so that the estimated number of infections roughly matches observed cases.

### From data to signal

Raw concentrations are first normalised by the daily flow, which removes much of the noise that a varying flow introduces into the raw signal. The normalised signal is then understood as a delayed and blurred trace of the infection process upstream: infected individuals begin to shed the pathogen some time after infection and continue to shed over a period of days, and the shed load is further blurred and delayed as it travels through the sewer network to the sampling site.

### Assumptions

To turn concentrations back into transmission dynamics we need a set of disease-specific distributions, typically taken from the literature: the generation time (the time between a primary infection and its secondary infections), the shedding load distribution (how much an average individual sheds over time), and, because the shedding load is usually expressed relative to symptom onset, the incubation period (the time between infection and symptom onset). All of these are discretised to a daily time step for use in the model.

### The latent model

Underlying the observations is a latent epidemic process. The number of new infections each day evolves according to a renewal process driven by a time-varying effective reproduction number, R_t. The reproduction number itself is allowed to vary smoothly over time, using a flexible smoothing model such as a Gaussian process or a random walk, so that we do not need to specify the trajectory in advance. The infection process also includes stochastic noise, capturing additional variability beyond what a deterministic renewal would imply.

### The observation model

The latent infections are mapped to an expected load by convolving the infection process with the shedding load distribution, and that expected load is in turn delayed through the sewer residence time before being divided by the flow to give an expected concentration. The observed concentrations are then modelled as this expected concentration corrupted by measurement noise. Together these pieces connect the observed wastewater concentrations to the hidden infection dynamics we want to estimate.

### Estimation

Combining the latent model and the observation model gives a full Bayesian generative model. Inference is performed with Hamiltonian MCMC sampling, which produces a full posterior distribution over every parameter. From that posterior we obtain R_t and the other transmission indicators — along with infections, expected load and concentration — together with credible intervals that quantify the uncertainty in every estimate.

## Derived from EpiSewer

This package is a Julia port of the [EpiSewer](https://github.com/adrian-lison/EpiSewer) R package by Adrian Lison and colleagues. The original model is described in:

> Lison, A., McLeod, R.E., Huisman, J.S. et al. Real-Time Estimation of Pathogen Transmission Dynamics from Wastewater. *Nature Communications* (2026). DOI: [10.1038/s41467-026-75380-3](https://doi.org/10.1038/s41467-026-75380-3)

This Julia version uses composable model components from the EpiAware.org ecosystem (`ComposableTuringIDModels.jl`, `CensoredDistributions.jl`, `EpiAwareADTools.jl`) rather than the Stan-based approach of the original.

## Model components

The model is composed of six modular components:

- **Measurements**: concentration observation model, noise estimation, limit-of-detection handling
- **Sampling**: outlier detection, sample batch effects
- **Sewage**: flow normalization, sewer residence time distributions
- **Shedding**: incubation period, shedding load distributions, load-per-case calibration, shedding variation
- **Infections**: generation time distribution, R_t estimation (GP, RW, splines), seeding, infection noise
- **Forecast**: probabilistic forecasts of R_t, infections, and concentrations

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
prior = sample(mdl_t, Prior(), 100; progress = false)
draw = first(vec(returned(mdl_t, prior)))
map(x -> round.(extrema(x); sigdigits = 3),
    (R_t = exp.(draw.Z_t), I_t = draw.I_t, concentration = draw.generated_y_t))
```

Summarise `R_t` across the prior draws.
`Renewal` samples a latent `Z_t` and applies its `exp` transformation, so `R_t = exp.(Z_t)`.

```julia
using Statistics: median

Rt = reduce(hcat, [exp.(g.Z_t) for g in vec(returned(mdl_t, prior))])
Rt_median = [median(view(Rt, i, :)) for i in axes(Rt, 1)]
(median = round(median(Rt_median); digits = 2),
    range = round.(extrema(Rt_median); digits = 2))
```

### Fitting it

Everything above runs in seconds, which is why it is the example.
Posterior inference on this model does not.
33 measurements inform 164 random-walk steps plus 87 imputed missing days.
The sampler adapts to a step size around 0.001 and then spends most of its time in maximum-depth trees.
Measured here, 2 chains of 300 warmup and 300 draws took 19 minutes and returned a maximum R-hat of 2.8 with a minimum effective sample size of 2, and 4 chains of 400 warmup and 400 draws did not finish inside 30 minutes.
The fit is therefore shown rather than run, and the block below is deliberately not executed as part of this page.

Two settings matter more than the iteration counts.
Do not leave the AD backend at Turing's default.
This is a few-hundred-parameter latent process, so reverse mode is far cheaper per gradient.
Measured on this model, Mooncake costs 1.4 ms per gradient against ForwardDiff's 24 ms.
Raise `target_accept` as well, because the default leaves divergent transitions behind.

```jl
import Mooncake

chn = sample(
    mdl_t, NUTS(0.9; adtype = Turing.AutoMooncake()),
    MCMCThreads(), 400, 4; warmup = 400, progress = false)

Rt = reduce(hcat, [exp.(g.Z_t) for g in vec(returned(mdl_t, chn))])
```

Budget tens of minutes.
Check `MCMCChains.gelmandiag` and the divergence count before reading anything off the result.
Treat `R_t` near the end of the series with care.
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

Full documentation is hosted at [seabbs.github.io/EpiSewer.jl](https://seabbs.github.io/EpiSewer.jl/stable/), including the model-components table and a page recording the LLM-assisted development process.

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
