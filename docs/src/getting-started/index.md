# [Getting started](@id getting-started)

The home page is generated from the README and carries the install instructions
and a worked example of the whole model.
This page introduces the three things you need to work with the package: the
entry point, the components it adds to the ecosystem, and how to swap them.

Every code block here runs when the documentation is built.

## One entry point

There is a single front end, `EpiSewer.model`.
It is public but not exported, so call it qualified.
It returns a `ComposableTuringIDModels.IDModel`, which is the ecosystem's
standard model object, so nothing downstream is specific to this package.

```@example gs
using EpiSewer
import ComposableTuringIDModels as CT

idm = EpiSewer.model()
typeof(idm)
```

The model is the infection process paired with an observation chain, and both
are reachable as fields.
Reading the observation chain from the outside in gives the order the
transformations are applied.

```@example gs
(
    infection = typeof(idm.infection_model).name.name,
    observation = typeof(idm.observation_model).name.name,
)
```

Turning it into a Turing model needs the observed series, the daily flow, and the
length of the infection series.
That last one is longer than the observation series, because each convolution in
the chain consumes a lead-in that `EpiSewer.observation_lead_in` reports.

```@example gs
d = EpiSewer.example_data()
y = d.measurements.concentration
flow = Vector{Float64}(d.flows.flow)

lead_in = EpiSewer.observation_lead_in(idm)
n = length(y) + lead_in
(observations = length(y), lead_in = lead_in, n = n)
```

Passing `n = length(y)` instead would leave the first `lead_in` observations
out of the likelihood without saying so.

## The components this package adds

Most of the model is existing ecosystem components.
The five below are the wastewater-specific pieces, and each is shown here on a
short series so you can see what it does to the expected values.

### `LogNormalError`

Relative measurement noise: the standard deviation is proportional to the
expected concentration, so its parameter is a coefficient of variation.
That is the right shape for concentrations spanning orders of magnitude.

```@example gs
using Distributions: mean, std

d_lne = EpiSewer.observation_error(EpiSewer.LogNormalError(), 100.0, 0.2)
(mean = mean(d_lne), sd = std(d_lne), cv = std(d_lne) / mean(d_lne))
```

### `LOD`

Left-censoring at a limit of detection.
A measurement reported at the limit scores the probability of being anywhere at
or below it, so it contributes far less than a density would.

```@example gs
using Distributions: logpdf, Normal

lod = EpiSewer.LOD(CT.NormalError(; std = Normal(10.0, 0.0)); lod = 50.0)
err = EpiSewer.observation_error(lod, 100.0, 10.0)
(at_the_limit = logpdf(err, 50.0), above_it = logpdf(err, 120.0))
```

### `MeasurementOutliers`

Independent spikes added to the expected series, drawn from a generalised
extreme value distribution truncated at zero.
The tail is extreme by design: most days are untouched and a rare day absorbs a
large excursion.

```@example gs
using Distributions: quantile

m_out = EpiSewer.MeasurementOutliers(CT.NormalError(); scale = 0.68)
(median_spike = quantile(m_out.spike, 0.5), q99 = quantile(m_out.spike, 0.99))
```

### `DigitalPCRError`

Scores the positive partition counts directly rather than a concentration
derived from them, through the Poisson partition law.

```@example gs
Y_log_copies = log.([0.01, 0.02, 0.04])
p = 1.0 .- exp.(-exp.(Y_log_copies))
(log_copies_per_partition = Y_log_copies, positive_probability = p)
```

### `FlowNormalize`

Divides the expected load by the daily flow to give an expected concentration,
which is what removes flow-driven dilution from the signal.

```@example gs
fn = EpiSewer.FlowNormalize(CT.NormalError())
load = fill(2.0e13, 3)
q = [2.5e11, 3.0e11, 2.0e11]
load ./ q
```

## Prior predictive draws

A model with no observations simulates from its prior, which is the quickest
check that a chain is assembled the way you intended.

```@example gs
using Random: MersenneTwister

sim = CT.as_turing_model(idm, (y = missing, flow = flow), n)(MersenneTwister(1))
(
    infections = round.(sim.I_t[1:5]; digits = 1),
    concentrations = round.(Float64.(sim.generated_y_t[(end - 4):end]); digits = 1),
)
```

## Swapping components

`infection_model` and `observation_model` are the swap points.
Override one and the other keeps its default.

```@example gs
using Distributions: Normal as Nrm

direct = CT.DirectInfections(; Z = CT.RandomWalk(), initialisation = Nrm())
swapped = EpiSewer.model(infection_model = direct)
(
    infection = typeof(swapped.infection_model).name.name,
    observation_unchanged = typeof(swapped.observation_model) ===
        typeof(idm.observation_model),
)
```

The delay inputs are keywords too, and each accepts a continuous distribution, an
already discretised PMF vector, or a prior model whose parameters are inferred.
A `nothing` delay omits that convolution, which is the default for the sewer
residence time.

```@example gs
using Distributions: Gamma, truncated

with_residence = EpiSewer.model(
    residence_dist = truncated(Gamma(2.0, 0.5), nothing, 5.0), D_residence = 5.0
)
(
    lead_in_default = EpiSewer.observation_lead_in(idm),
    lead_in_with_residence = EpiSewer.observation_lead_in(with_residence),
)
```

The lead-in grows because the extra convolution consumes more of the series.

## Learning more

- The [model components](@ref) page maps every EpiSewer component onto its
  counterpart here, and records what the default chain leaves out.
- The [Public API](@ref public-api) lists the full interface.
- To report a problem or ask a question, open an issue or a discussion on the
  [GitHub repository](https://github.com/seabbs/EpiSewer.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
