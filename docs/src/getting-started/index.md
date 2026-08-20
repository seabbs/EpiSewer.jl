# [Getting started](@id getting-started)

The home page runs one worked example end to end, from the example data to the
fitted model.
Fitting a series of your own almost always means changing something in that
model, and changing it means knowing how it was put together.
This page covers the assembly: the entry point, the two places a whole stage
can be replaced, and what a replacement does to the model.

## The one entry point

The package has a single front end, `EpiSewer.model`, which takes the disease
assumptions and returns the assembled model.

```@example gs
using EpiSewer, ComposableTuringIDModels

assumptions = EpiSewer.example_assumptions()
idm = EpiSewer.model(; assumptions...)
(name = nameof(typeof(idm)), owner = parentmodule(typeof(idm)))
```

The return value is a `ComposableTuringIDModels.IDModel`, the ecosystem's
standard model object, so everything downstream of `model()` is ecosystem
machinery rather than package-specific code.
`as_turing_model` turns it into a Turing model, and `forecast`, `spread_draws`
and the rest of the ecosystem's tooling apply unchanged.

## The two swap points

An `IDModel` pairs an infection process with an observation chain, and both are
fields.
`model()`'s `infection_model` and `observation_model` keywords set them
directly, which is what makes them the swap points, because passing either
replaces that whole stage.
Every other keyword tunes a piece of the default the package builds when you do
not.

```@example gs
(
    infection = nameof(typeof(idm.infection_model)),
    rt_process = nameof(typeof(idm.infection_model.rt)),
    renewal_step = nameof(typeof(idm.infection_model.recurrent_step)),
    observation = nameof(typeof(idm.observation_model)),
)
```

The observation model reports only its outermost wrapper, because the chain is
a nesting of modifiers around an error model.
Each wrapper holds the model it wraps in a field, so walking those fields gives
the chain from the outside in, in the order the transformations are applied.

```@example gs
function observation_chain(m)
    stages = [nameof(typeof(m))]
    for f in fieldnames(typeof(m))
        inner = getfield(m, f)
        if inner isa AbstractObservationModel
            append!(stages, observation_chain(inner))
            break
        end
    end
    return stages
end

observation_chain(idm.observation_model)
```

The [Public API](@ref public-api) documents each stage this package adds, and
the [Model components](@ref model-components) page maps them onto the R model.

## Sizing the infection series

Turning the assembly into a Turing model needs the observed series, the daily
flow, and `n`, the length of the infection series.
Each `LatentDelay` in the chain drops the partially observed head of its
convolution, so `n` has to exceed the number of observations by the total the
delays consume.
`EpiSewer.observation_lead_in` reports that total.

```@example gs
d = EpiSewer.example_data()
y = d.measurements.concentration
flow = Vector{Float64}(d.flows.flow)

lead_in = EpiSewer.observation_lead_in(idm)
n = length(y) + lead_in
(observations = length(y), lead_in = lead_in, n = n)
```

The observation-error model right-aligns the observations against whatever
expected series it is given, so passing `n = length(y)` silently leaves the
first `lead_in` observations unscored.
Taking `n` from `observation_lead_in` instead scores every observation.

## Drawing from the prior

What the priors imply is worth seeing before any measurement pulls on them.
Sampling the assembled model with `Prior()` and replaying it with `returned`
gives the latent series they generate on their own.
Two helpers cover the figures on this page: quantiles of a generated quantity
across draws, and median lines over their 50% intervals, one colour per model.

```@example gs
using Turing, DataFrames, AlgebraOfGraphics, CairoMakie
using Random: Xoshiro
using Statistics: quantile

function prior_draws(m, y_t, n_t; n_draws = 300, seed = 1)
    mdl = as_turing_model(m, y_t, n_t)
    chn = sample(Xoshiro(seed), mdl, Prior(), n_draws; progress = false)
    return vec(returned(mdl, chn))
end

function summarise(draws, f, label)
    m = reduce(hcat, [f(g) for g in draws])
    q(p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
    return DataFrame(
        t = axes(m, 1), med = q(0.5), lo50 = q(0.25), hi50 = q(0.75),
        model = label
    )
end

function series_plot(dfs...; kwargs...)
    layers = data(reduce(vcat, dfs)) * (
        mapping(:t, :lo50, :hi50, color = :model) * visual(Band; alpha = 0.3) +
            mapping(:t, :med, color = :model) * visual(Lines; linewidth = 2)
    )
    return draw(layers; axis = (; kwargs...), figure = (; size = (860, 300)))
end
```

The default `R_t` prior is the R package's pair of approximate Gaussian
processes, summed under a softplus link.

```@example gs
y_t = (y = y, flow = flow)
draws = prior_draws(idm, y_t, n)
rt_prior = summarise(draws, g -> exp.(g.Z_t), "default")

series_plot(rt_prior; ylabel = "R_t", xlabel = "day")
```

The infections those draws imply span orders of magnitude, which is what a
prior on transmission with no data to constrain it looks like on a series this
long.

```@example gs
series_plot(
    summarise(draws, g -> g.I_t, "default");
    yscale = log10, ylabel = "infections", xlabel = "day"
)
```

## Swapping in a component

An `IDModel` and everything inside it are plain structs, so a component can be
replaced on an assembled model instead of rebuilt from the assumptions.
`Accessors.@set` returns a copy with one field changed.

```@example gs
using Accessors: @set

walk = @set idm.infection_model.rt = RandomWalk()
```

`EpiSewer.model(; assumptions..., rt = RandomWalk())` reaches the same model
from the assumptions.
`@set` is the shorter route once a model is in hand, and it is the only route
for a field no keyword exposes.

Either way the swapped model goes through `as_turing_model` exactly as the
default does, so the consequence of the swap can be read off its prior.

```@example gs
series_plot(
    rt_prior,
    summarise(prior_draws(walk, y_t, n), g -> exp.(g.Z_t), "random walk");
    ylabel = "R_t", xlabel = "day"
)
```

The random walk's interval is wider from the start and keeps widening as the
series runs on, because nothing pulls it back, whereas the Gaussian processes
hold their length scale.
Part of the gap is the link rather than the process, because the default `rt`
puts a softplus on the latent path while a bare latent model keeps the
exponential `Renewal` applies.
[Adapting the model](@ref adapting-the-model) separates the two.

## Learning more

- [Adapting the model](@ref adapting-the-model) retunes the defaults for a new
  series and works through the `R_t` process, the delays and the measurement
  model.
- [Case surveillance](@ref case-surveillance) puts a case observation chain
  alongside the wastewater one and fits both on the same infections.
- The [Model components](@ref model-components) page maps every EpiSewer
  component onto the ecosystem piece that provides it.
- The [Public API](@ref public-api) lists the full interface.
- To report a problem or ask a question, open an issue or a discussion on the
  [GitHub repository](https://github.com/seabbs/EpiSewer.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
