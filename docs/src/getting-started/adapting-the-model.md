# [Adapting the model](@id adapting-the-model)

The default model carries one set of choices about transmission, shedding and
measurement.
A different site, a different pathogen, or a different laboratory protocol
changes some of them.
This page works through those changes on a series the defaults were not built
for, and reads the consequence off the model's prior.
Nothing here is fitted: what an assumption does to the model is visible in what
the model generates before it sees a measurement.

## Retuning for a new series

The last 60 days of the Zurich example series stand in for a series the
defaults were not tuned to.

```@example adapt
using EpiSewer, DataFrames, DataFramesMeta, Turing
import ComposableTuringIDModels as CT
using Distributions: Normal, Gamma, truncated
using Statistics: median, quantile, std
using Random: Xoshiro

d = EpiSewer.example_data()
assumptions = EpiSewer.example_assumptions()

n_days = 60
y = last(d.measurements, n_days).concentration
flow = Vector{Float64}(last(d.flows.flow, n_days))
(
    days = length(y),
    measured = count(!ismissing, y),
    concentration = round.(extrema(skipmissing(y)); digits = 1),
)
```

The disease assumptions travel with the pathogen, so `example_assumptions`
still applies.
Three of `model()`'s remaining defaults are properties of the series rather
than of the disease, and each is read off the data.

```@example adapt
idm = EpiSewer.model(; assumptions...)
lead_in = EpiSewer.observation_lead_in(idm)
n = length(y) + lead_in

lpc = exp(median(assumptions.lpc_prior))
retune = (
    initial_infections = EpiSewer.crude_initial_infections(y, flow, lpc),
    n_gp = n,
    outlier_scale = lpc / median(flow),
)
(
    initial_infections = round(retune.initial_infections; digits = 1),
    n_gp = retune.n_gp,
    outlier_scale = round(retune.outlier_scale; sigdigits = 4),
)
```

`initial_infections` centres the seeding prior, and
`crude_initial_infections` converts the concentrations measured in the first
week into a case count at the assumed load per case.
Here it is 1881 against the shipped 913, because this window starts partway
through an epidemic rather than at the start of the series.
`outlier_scale` is the concentration equivalent of one unit of outlier spike,
the load per case over the median flow, so it moves with the site's flow.
`n_gp` is the series length the Gaussian-process length scale is converted at.

```@example adapt
(
    default_scale = round(EpiSewer.gp_length_scale(21.0, 164); digits = 3),
    matched_scale = round(EpiSewer.gp_length_scale(21.0, n); digits = 3),
    default_in_days = round(
        EpiSewer.gp_length_scale(21.0, 164) * std(1:n); digits = 1
    ),
)
```

`HilbertSpaceGP` measures its length scale in standard deviations of the time
index, and the index of a 104-day series has a smaller standard deviation than
that of the 164-day one the default assumes.
Carrying the default over therefore states a length scale of 13.3 days on this
series rather than the 21 days it was taken from.

Sampling with `Prior()` and replaying with `returned` gives the latent series
the priors imply.
Three helpers cover the figures below: individual draws of a generated
quantity, quantiles of it across draws, and median lines over their 50%
intervals.

```@example adapt
using AlgebraOfGraphics, CairoMakie

function prior_draws(m, y_t, n_t; n_draws = 200, seed = 1)
    mdl = CT.as_turing_model(m, y_t, n_t)
    chn = sample(Xoshiro(seed), mdl, Prior(), n_draws; progress = false)
    return vec(returned(mdl, chn))
end

function paths(draws, f, label; k = 6)
    return reduce(
        vcat,
        [
            DataFrame(
                t = eachindex(f(g)), value = f(g), draw = i, model = label
            ) for (i, g) in enumerate(draws[1:k])
        ]
    )
end

function summarise(draws, f, label)
    m = reduce(hcat, [f(g) for g in draws])
    q(p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
    return DataFrame(
        t = axes(m, 1), med = q(0.5), lo50 = q(0.25), hi50 = q(0.75),
        model = label
    )
end

function paths_plot(dfs...; kwargs...)
    layers = data(reduce(vcat, dfs)) *
        mapping(:t, :value, group = :draw => nonnumeric, layout = :model) *
        visual(Lines; linewidth = 1.2)
    return draw(layers; axis = (; kwargs...), figure = (; size = (860, 300)))
end

function series_plot(dfs...; kwargs...)
    layers = data(reduce(vcat, dfs)) * (
        mapping(:t, :lo50, :hi50, color = :model) * visual(Band; alpha = 0.3) +
            mapping(:t, :med, color = :model) * visual(Lines; linewidth = 2)
    )
    return draw(layers; axis = (; kwargs...), figure = (; size = (860, 300)))
end

y_t = (y = y, flow = flow)
rt_of(g) = exp.(g.Z_t)

tuned = EpiSewer.model(; assumptions..., retune...)
tuned_draws = prior_draws(tuned, y_t, n)
default_draws = prior_draws(idm, y_t, n)
nothing # hide
```

The length scale sets how fast a draw turns over, not how far it strays, so it
shows in the individual draws rather than in the marginal interval.

```@example adapt
paths_plot(
    paths(default_draws, rt_of, "n_gp = 164 (default)"),
    paths(tuned_draws, rt_of, "n_gp = $n (this series)");
    ylabel = "R_t", xlabel = "day"
)
```

```@example adapt
daily_change(x) = abs.(diff(log.(x)))
typical(draws) = round(
    median([median(daily_change(rt_of(g))) for g in draws]); sigdigits = 3
)
(default = typical(default_draws), matched = typical(tuned_draws))
```

The carried-over length scale moves `R_t` a third further each day than the
converted one.

## The reproduction number process

`rt` takes any latent model in the ecosystem.
The link matters as much as the process: `Renewal` exponentiates the path it is
given, and the default `rt` pre-applies `EpiSewer.softplus_link` so that the
exponential is undone and `R_t` grows linearly in the latent path.
A bare latent model keeps the exponential.

```@example adapt
walk = CT.RandomWalk(;
    init = Normal(0.0, 0.2),
    ϵ_t = CT.HierarchicalNormal(; std = truncated(Normal(0, 0.05), 0, Inf))
)
autoreg = CT.AR(;
    damp = truncated(Normal(0.9, 0.05), 0, 1), init = Normal(0.0, 0.2),
    ϵ_t = CT.HierarchicalNormal(; std = truncated(Normal(0, 0.05), 0, Inf))
)
link(process) = CT.TransformLatentModel(process, EpiSewer.softplus_link)

walk_model = EpiSewer.model(; assumptions..., retune..., rt = link(walk))
ar_model = EpiSewer.model(; assumptions..., retune..., rt = link(autoreg))
exp_model = EpiSewer.model(; assumptions..., retune..., rt = walk)

walk_draws = prior_draws(walk_model, y_t, n)
ar_draws = prior_draws(ar_model, y_t, n)
exp_draws = prior_draws(exp_model, y_t, n)

series_plot(
    summarise(tuned_draws, rt_of, "Gaussian processes"),
    summarise(walk_draws, rt_of, "random walk"),
    summarise(ar_draws, rt_of, "AR(1)");
    ylabel = "R_t", xlabel = "day"
)
```

The random walk starts at the spread of its `init` prior and widens as the
series runs on, because nothing pulls it back.
The autoregressive process reverts to its mean, so its interval settles at the
stationary spread its damping and innovation priors imply.
The Gaussian processes hold a constant width, which is the marginal standard
deviation their magnitude priors state.

The two links differ in the upper tail, where the same latent path is either
exponentiated or passed through a softplus.

```@example adapt
last_rt(draws, p) = round.(
    quantile([exp(g.Z_t[end]) for g in draws], p); sigdigits = 3
)
(
    quantile = [0.5, 0.9, 0.99],
    softplus = last_rt(walk_draws, [0.5, 0.9, 0.99]),
    exponential = last_rt(exp_draws, [0.5, 0.9, 0.99]),
)
```

The exponential link puts `R_t` above 7 once in a hundred draws on the last day
of the series, against 3 under the softplus.

## Removing components

Three keywords take `nothing` and drop a stage: `infection_noise` for
stochastic infections, `load_cv` for individual-level load variation, and
`outlier_scale` for the spike component.

```@example adapt
variants = [
    "default" => tuned,
    "no infection noise" => EpiSewer.model(;
        assumptions..., retune..., infection_noise = nothing
    ),
    "no load variation" => EpiSewer.model(;
        assumptions..., retune..., load_cv = nothing
    ),
    "no outlier spikes" => EpiSewer.model(;
        assumptions..., retune..., outlier_scale = nothing
    ),
]
map(v -> nameof(typeof(last(v).infection_model.recurrent_step)), variants)
```

Dropping the infection noise makes the renewal step deterministic, and the
other two removals act in the observation chain.
Each takes a source of day-to-day variability out of the generated series.
Two statistics per draw locate them: the typical daily change in infections,
and the largest daily jump in the expected concentration.

```@example adapt
removals = reduce(
    vcat,
    map(variants) do (label, m)
        draws = label == "default" ? tuned_draws : prior_draws(m, y_t, n)
        DataFrame(
            model = label,
            infections = [median(daily_change(g.I_t)) for g in draws],
            concentration = [
                maximum(daily_change(g.expected_y_t)) for g in draws
            ],
        )
    end
)
long = @chain removals begin
    stack([:infections, :concentration];
        variable_name = :series, value_name = :value)
    @rtransform :series = :series == "infections" ?
        "infections: typical daily change" :
        "concentration: largest daily jump"
end

draw(
    data(long) * mapping(:model, :value, layout = :series) *
        visual(BoxPlot; width = 0.55);
    axis = (;
        yscale = log10, xticklabelrotation = 0.35,
        ylabel = "absolute daily log change"
    ),
    facet = (; linkyaxes = :none), figure = (; size = (900, 380))
)
```

Dropping the infection noise halves the daily change in infections: what is
left is the movement `R_t` alone implies.
Dropping the spikes removes the upper tail of the daily jump in concentration
entirely, so the largest jump in a draw becomes the largest jump transmission
can produce.

Dropping the load variation moves neither, because its size is set by how many
people are shedding.
The realised load is a sum over shedding individuals, so its relative spread is
the coefficient of variation over the square root of that number.

```@example adapt
load_variation = EpiSewer.LoadVariation(
    EpiSewer.LogNormalError(; cv = Normal(0.001, 0.0)); cv = 1.0
)
map([10.0, 1.0e3, 1.0e5]) do shedding
    g = CT.as_turing_model(
        load_variation, missing, fill(shedding, 400)
    )(Xoshiro(2))
    (
        shedding = shedding,
        relative_sd = round(
            std(g.expected) / shedding; digits = 4
        ),
    )
end
```

Ten shedding individuals give a 30% spread on the load, a hundred thousand give
0.3%.
This series sits at the second end, so removing the component costs nothing
here and matters for a small catchment or an early outbreak.

## Delay inputs

`generation_time`, `shedding_dist`, `incubation_dist` and `residence_dist` each
take a continuous `Distribution`, an already discretised PMF vector, or a prior
model whose parameters are inferred.
The shedding load profile is the longest of them, so it is where the choice
costs the most.

```@example adapt
pmf_of(dist, D) = reverse(
    CT.LatentDelay(EpiSewer.LogNormalError(), dist; D = D).delay
)
shed_38 = pmf_of(assumptions.shedding_dist, 38.0)
shed_21 = pmf_of(assumptions.shedding_dist, 21.0)

uncertain_shedding = CT.UncertainDelay(
    Gamma,
    [
        truncated(Normal(0.93, 0.15), 0, Inf),
        truncated(Normal(7.24, 1.5), 0, Inf),
    ];
    D = 38.0, Δd = 1.0
)

forms = (
    continuous = EpiSewer.model(; assumptions..., retune...),
    discretised = EpiSewer.model(;
        assumptions..., retune..., shedding_dist = shed_21
    ),
    inferred = EpiSewer.model(;
        assumptions..., retune..., shedding_dist = uncertain_shedding
    ),
)
map(EpiSewer.observation_lead_in, forms)
```

The lead-in follows the PMF length.
A profile truncated at 21 days rather than 38 costs 17 fewer days of infection
series for the same measurements, and the mass beyond the horizon is
redistributed over the days that remain.
The inferred delay holds its length through its fixed horizon `D`, so it costs
what the continuous form costs, and its shape is drawn each iteration.

```@example adapt
pmf_frame(p, form; draw_id = 1) = DataFrame(
    day = 0:(length(p) - 1), mass = p, draw = draw_id, form = form
)

mdl = CT.as_turing_model(uncertain_shedding)
chn = sample(Xoshiro(3), mdl, Prior(), 40; progress = false)
drawn = reduce(
    vcat,
    [
        pmf_frame(p, "inferred"; draw_id = i)
            for (i, p) in enumerate(vec(returned(mdl, chn)))
    ]
)
fixed = vcat(
    pmf_frame(shed_38, "continuous, D = 38"),
    pmf_frame(shed_21, "discretised, D = 21"),
)

draw(
    data(drawn) * mapping(:day, :mass, group = :draw => nonnumeric,
        color = :form) * visual(Lines; alpha = 0.35, linewidth = 1) +
        data(fixed) * mapping(:day, :mass, color = :form) *
        visual(Lines; linewidth = 2);
    axis = (;
        xlabel = "days since symptom onset", ylabel = "shed load (share)"
    ),
    figure = (; size = (760, 340))
)
```

## Measurement models

The measurement stage is the innermost part of the observation chain, so
swapping it means rebuilding the chain around it.
The stages are the same components the default assembles.

```@example adapt
function measurement_chain(error_model)
    obs = EpiSewer.FlowNormalize(
        EpiSewer.MeasurementOutliers(
            error_model; scale = retune.outlier_scale
        )
    )
    obs = CT.LatentDelay(obs, assumptions.shedding_dist; D = 38.0)
    obs = CT.Ascertainment(obs, assumptions.lpc_prior)
    obs = EpiSewer.LoadVariation(obs; cv = 1.0)
    return CT.LatentDelay(obs, assumptions.incubation_dist; D = 8.0)
end

lod = 200.0
lod_model = EpiSewer.model(;
    assumptions..., retune...,
    observation_model = measurement_chain(
        EpiSewer.LOD(EpiSewer.LogNormalError(); lod = lod)
    ),
)
(
    lead_in = EpiSewer.observation_lead_in(lod_model),
    default_lead_in = lead_in,
)
```

`LOD` left-censors the measurement at a detection limit, which is the model for
a series whose non-detects are reported at the limit rather than as missing
values.
A measurement model is not a delay, so the rebuilt chain carries the lead-in
the default does.
Passing `y = missing` leaves the measurements to be drawn rather than scored,
so each draw pairs an expected concentration with the measurement the model
would record.

```@example adapt
lod_draws = prior_draws(
    lod_model, (y = missing, flow = flow), n; n_draws = 60
)
measurements = DataFrame(
    expected = reduce(vcat, [g.expected_y_t for g in lod_draws]),
    drawn = reduce(vcat, [Float64.(g.generated_y_t) for g in lod_draws]),
)
(
    days = nrow(measurements),
    at_limit = count(==(lod), measurements.drawn),
)
```

```@example adapt
draw(
    data(@rsubset(measurements, 1.0 < :expected < 1.0e5)) *
        mapping(:expected, :drawn) *
        visual(Scatter; markersize = 4, alpha = 0.25) +
        data((; limit = [lod])) * mapping(:limit) *
        visual(HLines; color = :grey40, linestyle = :dash),
    axis = (;
        xscale = log10, yscale = log10,
        xlabel = "expected concentration (gc/mL)",
        ylabel = "drawn measurement (gc/mL)"
    ),
    figure = (; size = (640, 380))
)
```

Above the limit the measurement tracks the expected concentration with relative
noise, and below it every draw is reported at the limit.

`DigitalPCRError` scores the positive partition counts instead of a
concentration, which is what a digital PCR assay reports.
It takes the log expected copies per partition, so the chain converts the
concentration with the partition volume before it.

```@example adapt
partition_volume = 4.5e-4     # mL per partition
partitions = 25_000

count_model = EpiSewer.model(;
    assumptions..., retune...,
    observation_model = measurement_chain(
        CT.TransformObservationModel(
            EpiSewer.DigitalPCRError(fill(partitions, n_days)),
            Y -> log.(Y .* partition_volume)
        )
    ),
)
count_draws = prior_draws(
    count_model,
    (y = Vector{Union{Missing, Int}}(missing, n_days), flow = flow),
    n; n_draws = 20
)
g = first(count_draws)
(
    positive_probability = round.(g.expected_y_t[1:4]; digits = 4),
    positives = g.generated_y_t[1:4],
)
```

A partition tests positive when it receives at least one copy, so the
probability is the Poisson chance of a non-empty partition and it saturates as
the concentration grows.

```@example adapt
concentration = exp10.(range(0, 4.5, 60))
assay = EpiSewer.DigitalPCRError(fill(partitions, length(concentration)))
assay_draw = CT.as_turing_model(
    assay, missing, log.(concentration .* partition_volume)
)(Xoshiro(4))
assayed = DataFrame(
    concentration = concentration,
    probability = assay_draw.expected,
    drawn = assay_draw.y_t ./ partitions,
)

draw(
    data(assayed) * (
        mapping(:concentration, :probability) *
            visual(Lines; color = :grey50) +
            mapping(:concentration, :drawn) * visual(Scatter; markersize = 6)
    );
    axis = (;
        xscale = log10, xlabel = "concentration (gc/mL)",
        ylabel = "positive partitions (share)"
    ),
    figure = (; size = (640, 340))
)
```

Above about 10⁴ gc/mL almost every partition is positive, so the counts stop
separating one concentration from another and the assay has to be diluted.

## Learning more

- The [Getting started](@ref getting-started) page covers the components this
  package adds and the two swap points on `model()`.
- The [Model components](@ref model-components) page maps every EpiSewer
  component onto the ecosystem piece that provides it.
- The [Public API](@ref public-api) lists the full interface.
