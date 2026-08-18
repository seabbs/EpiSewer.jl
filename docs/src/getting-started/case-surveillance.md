# [Case surveillance](@id case-surveillance)

Wastewater concentrations and reported cases observe the same infections
through different instruments.
Each carries its own delays, its own reporting fraction, and its own noise.
`ComposableTuringIDModels.Split` puts both observation chains on one infection
process, so the two are fitted together.

## Data

The example series carries reported cases alongside the concentrations and the
daily flow, over the same 120 days.

```@example cases
using EpiSewer
import ComposableTuringIDModels as CT
using Distributions

d = EpiSewer.example_data()
y = d.measurements.concentration
flow = Vector{Float64}(d.flows.flow)
cases = [ismissing(c) ? missing : round(Int, c) for c in d.cases.cases]
(
    days = length(cases),
    measured = count(!ismissing, y),
    reported = count(!ismissing, cases),
)
```

The case series is rounded to counts, which is the scale a negative binomial
scores.

## The case observation chain

Reported cases are infections that have incubated, been ascertained, and then
been reported after a further delay.
The chain is assembled innermost first, so the outermost wrapper is the first
transformation applied to infections.

```@example cases
assumptions = EpiSewer.example_assumptions()

case_stream = CT.LatentDelay(
    CT.Ascertainment(
        CT.LatentDelay(
            CT.NegativeBinomialError(), Gamma(2.0, 2.0); D = 14.0
        ),
        Normal(log(0.2), 0.1)
    ),
    assumptions.incubation_dist; D = 8.0
)
```

Read it outside in.
The incubation period puts infections on the symptom-onset timescale, and it
is the same distribution the wastewater chain uses.
`Ascertainment` scales by the fraction of infections ever reported, on the log
scale, so this prior is centred on 20%.
The reporting delay carries onsets to their report date, and
`NegativeBinomialError` scores the counts.

## Composing the two streams

`Split` feeds one expected series to several named streams.
Placed on infections the streams are parallel, each observing the same `I_t`.
The wastewater chain is the one [`EpiSewer.model`](@ref EpiSewer.model)
assembles, taken off the default model and used unchanged.

```@example cases
idm = EpiSewer.model(; assumptions...)
joint = EpiSewer.model(;
    assumptions...,
    observation_model = CT.Split(
        (wastewater = idm.observation_model, cases = case_stream)
    )
)
joint.observation_model
```

Each stream is a branch of the printed tree, and each stream's sampled
variables are prefixed with its name.

## Sizing the infection series

Every `LatentDelay` drops the partially observed head of its convolution, so
the infection series has to run further back than the observations.
`EpiSewer.observation_lead_in` reports that total, and takes the longest branch
of a split.

```@example cases
lead_in = (
    wastewater = EpiSewer.observation_lead_in(idm),
    cases = EpiSewer.observation_lead_in(case_stream),
    joint = EpiSewer.observation_lead_in(joint),
)
```

The shedding load profile is the longest delay in either chain, so the
wastewater branch sets the length of the infection series.
The case chain is shorter, so it has an expected value for every day the
wastewater chain reaches and for the days before it as well.
Every expected value needs an entry in its observed series, so the case series
is padded with `missing` over the days that precede the record.

```@example cases
n = length(y) + lead_in.joint
pad = lead_in.joint - lead_in.cases
y_t = (
    wastewater = (y = y, flow = flow),
    cases = vcat(fill(missing, pad), cases),
)
(n = n, pad = pad, cases = length(y_t.cases))
```

Each stream reads its own entry from `y_t` by name.
The wastewater entry carries the concentrations together with the daily flow
its chain divides by.

## Prior predictive

Simulating from the prior needs no observations, so both series are passed as
`missing`.
The flow stays, because it is data rather than something the model generates.

```@example cases
using Turing, DataFrames, DataFramesMeta, AlgebraOfGraphics, CairoMakie
using Random: Xoshiro
using Statistics: quantile

mdl = CT.as_turing_model(
    joint, (wastewater = (y = missing, flow = flow), cases = missing), n
)
chn = sample(Xoshiro(1), mdl, Prior(), 300; progress = false)
draws = vec(returned(mdl, chn))
```

A stream's series is right-aligned on the infection series, so its first day is
read off its length.
Values before that are `missing`, and counts of zero are dropped because the
figures are on a log scale.

```@example cases
clean(x) = Float64.(collect(skipmissing(x)))
days(x) = (n - length(x) + 1):n

panels = ["infections", "concentration (gc/mL)", "reported cases"]
streams(g) = (
    panels[1] => g.I_t,
    panels[2] => g.generated_y_t.wastewater,
    panels[3] => g.generated_y_t.cases,
)
```

Each draw is one infection path and the two series it generates.

```@example cases
paths = @chain vcat(
        [
            DataFrame(
                t = days(clean(v)), value = clean(v), panel = k, draw = i
            ) for (i, g) in enumerate(draws[1:6]) for (k, v) in streams(g)
        ]...
    ) @rsubset :value > 0

draw(
    data(paths) * mapping(
        :t, :value, color = :draw => nonnumeric,
        col = :panel => sorter(panels...)
    ) * visual(Lines; linewidth = 1.5);
    axis = (; yscale = log10, xlabel = "day", ylabel = ""),
    facet = (; linkyaxes = :none),
    figure = (; size = (950, 260)),
    legend = (; show = false),
)
```

A draw that grows in one stream grows in the other, because both read the same
infections.
What differs is the timing and the noise: the case series is smoother, and it
starts earlier because its delays are shorter.

Across draws, the interval on the expected series spans orders of magnitude,
which is what a prior on transmission with nothing scored against it gives.

```@example cases
function prior_interval(f, label)
    m = reduce(hcat, [clean(f(g)) for g in draws])
    q(p) = [quantile(view(m, i, :), p) for i in axes(m, 1)]
    return DataFrame(
        t = days(view(m, :, 1)), med = q(0.5), lo = q(0.25), hi = q(0.75),
        panel = label
    )
end

expected = vcat(
    prior_interval(g -> g.expected_y_t.wastewater, panels[2]),
    prior_interval(g -> g.expected_y_t.cases, panels[3])
)
observed = @chain vcat(
        DataFrame(t = days(y), obs = y, panel = panels[2]),
        DataFrame(t = days(cases), obs = cases, panel = panels[3])
    ) begin
    @rsubset !ismissing(:obs)
    @rtransform :obs = Float64(:obs)
end

facets = sorter(panels[2], panels[3])
layers = data(expected) * (
    mapping(:t, :lo, :hi, col = :panel => facets) *
        visual(Band; alpha = 0.3) +
        mapping(:t, :med, col = :panel => facets) *
        visual(Lines; linewidth = 2)
) + data(observed) * mapping(:t, :obs, col = :panel => facets) *
    visual(Scatter; markersize = 4, color = :black)

draw(
    layers;
    axis = (; yscale = log10, xlabel = "day", ylabel = ""),
    facet = (; linkyaxes = :none),
    figure = (; size = (860, 280)),
)
```

The case panel begins 24 days before the wastewater panel, which is the
difference in lead-in between the two chains.

Passing `y_t` in place of the two `missing` series conditions on both, and the
joint model then samples as the single-stream one does.
