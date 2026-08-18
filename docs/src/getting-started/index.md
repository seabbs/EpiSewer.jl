# [Getting started](@id getting-started)

The home page runs one worked example end to end, from the example data to the
fitted model.
This page is about composition: the entry point, the components this package
adds to the ecosystem, and how to customise the assembly.

## The one entry point

There is a single front end, `EpiSewer.model`.
It takes the disease assumptions and returns the assembled model.

```@example gs
using EpiSewer
import ComposableTuringIDModels as CT

assumptions = EpiSewer.example_assumptions()
idm = EpiSewer.model(; assumptions...)
(name = nameof(typeof(idm)), owner = parentmodule(typeof(idm)))
```

The return value is a `ComposableTuringIDModels.IDModel`, the ecosystem's
standard model object, so everything downstream of `model()` is ecosystem
machinery rather than package-specific code.
`as_turing_model` turns it into a Turing model, and `forecast`, `spread_draws`
and the rest of the ecosystem's tooling apply unchanged.

### The two swap points

An `IDModel` pairs an infection process with an observation chain, and both are
fields.
`model()`'s `infection_model` and `observation_model` keywords set them
directly, which is what makes them the swap points: pass either and that whole
stage is yours.
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
the chain.

```@example gs
function observation_chain(m)
    stages = [nameof(typeof(m))]
    for f in fieldnames(typeof(m))
        inner = getfield(m, f)
        if inner isa CT.AbstractObservationModel
            append!(stages, observation_chain(inner))
            break
        end
    end
    return stages
end

observation_chain(idm.observation_model)
```

Read that list outside in and it is the order the transformations are applied.
The default chain convolves infections with the incubation period, varies the
load shed between individuals, scales by the load shed per case, convolves with
the shedding load profile, divides by the daily flow, adds outlier spikes, and
scores the result with relative log-normal noise.

### Sizing the infection series

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
expected series it is given.
Passing `n = length(y)` therefore leaves the first `lead_in` observations
unscored, with nothing to signal it.
Read `n` back from `observation_lead_in` and every observation is scored.

## Drawing from the prior

Sampling the assembled model with `Prior()` and replaying it with `returned`
gives the latent series the priors imply, before any measurement has been
scored.
Two helpers cover the figures on this page: quantiles of a generated quantity
across draws, and median lines over their 50% intervals, one colour per model.

```@example gs
using Turing, DataFrames, DataFramesMeta, AlgebraOfGraphics, CairoMakie
using Random: Xoshiro
using Statistics: quantile

function prior_draws(m, y_t, n_t; n_draws = 300, seed = 1)
    mdl = CT.as_turing_model(m, y_t, n_t)
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
draws = prior_draws(idm, (y = y, flow = flow), n)
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

## The components this package adds

Seven components cover the wastewater-specific parts of the model.
Each one below is shown on a short series, so you can see what it does to the
expected values and to a draw.

Six of them are observation models, and a shared helper draws from any of
them.

```@example gs
prior_draw(m, Y_t, y_t = missing; seed = 1) =
    CT.as_turing_model(m, y_t, Y_t)(Xoshiro(seed))
```

The helper builds the Turing model for a bare component and evaluates it once
with no observations, which draws from the prior.
It returns the expected series the component produces and the values drawn
against it.

### `InfectionNoise`

Stochastic infections, on the infection side rather than the observation side.
This is an `AbstractRenewalModifier`, so `model()` composes it onto the renewal
step rather than into the observation chain, and it is what makes the default
renewal step a `RenewalStep` instead of a `ConstantRenewalStep`.

A deterministic renewal process fixes infections at the renewal expectation.
The modifier replaces that with a draw whose variance is the negative-binomial
variance at the same mean, so infections carry variance beyond what `R_t`
implies.
The draw, not the expectation, is what the next day convolves, so the noise
compounds through the process.

```@example gs
noise = EpiSewer.InfectionNoise()
ξ = noise.overdispersion
expectations = [10.0, 100.0, 1000.0, 10000.0]
sds = sqrt.(expectations .* (1 .+ expectations .* ξ^2))
(
    overdispersion = ξ,
    expectation = expectations,
    sd = round.(sds; digits = 1),
    relative_sd = round.(sds ./ expectations; digits = 3),
)
```

Poisson variance dominates at low incidence and the overdispersion dominates at
high, so the relative spread falls towards `ξ` as infections grow.

The relative spread carries a soft upper limit, because it runs the other way
too: it diverges as the expectation approaches zero, which would give an
arbitrarily small expectation an arbitrarily wide draw.

```@example gs
tiny = 1.0e-6
(
    cv_cap = noise.cv_cap,
    relative_sd_here = round.(sds ./ expectations; digits = 3),
    relative_sd_at_1e_6 = round(sqrt(1 / tiny + ξ^2); digits = 1),
)
```

Every expectation in the table sits well below the limit, so it barely bites and
the moments above hold to within a few per cent.
At an expectation of `1e-6` the uncapped relative spread would be the number on
the last line, and the limit is what holds it to `cv_cap` instead.

The family the moments are matched to is an argument, so the draw can be any
distribution the ecosystem can reparameterise by its mean and standard
deviation.
The default is a `LogNormal`, which keeps infections positive whatever the
sampler proposes; `dist = Normal` gives the R package's linear form.

```@example gs
(default_family = EpiSewer.InfectionNoise().dist,)
```

The parameterisation is non-centred: the sampled quantity is a standard normal
per day, and the location and scale are applied afterwards.
`InfectionNoiseDraws` is what the modifier resolves to once those normals are
drawn, and it is what the renewal scan steps through.

```@example gs
using Random: randn

resolved = EpiSewer.InfectionNoiseDraws(
    randn(Xoshiro(1), length(expectations)),
    noise.dist, ξ, noise.cv_cap, noise.cv_sharpness,
)

function noisy_infections(mod, expectations)
    state = 0
    drawn = similar(expectations)
    for (i, ι) in enumerate(expectations)
        drawn[i], state = CT.apply_modifier(mod, ι, state)
    end
    return drawn
end

round.(noisy_infections(resolved, expectations); digits = 1)
```

### `LoadVariation`

Variation in the load shed between individuals.
The series it takes counts shedding individuals, each of whom sheds a random
load, so the realised load is a sum of that many individual draws rather than
that many times a fixed amount.
Summing gamma loads with coefficient of variation `cv` keeps the expected value
and adds variance proportional to it, so the relative spread falls as the
number of shedders grows.

```@example gs
using Distributions: Normal, mean, std

lv = EpiSewer.LoadVariation(EpiSewer.LogNormalError(; cv = Normal(0.01, 0.0)))
varied = prior_draw(lv, fill(400.0, 500)).expected
(
    shedders = 400.0,
    mean_load = round(mean(varied); digits = 1),
    sd_load = round(std(varied); digits = 1),
    sd_predicted = round(sqrt(400.0); digits = 1),
)
```

At the default `cv = 1` that is Poisson-equivalent variance, which is what
EpiSewer fixes it to.

### `FlowNormalize`

Divides the expected load by the daily flow to give an expected concentration,
which is what removes flow-driven dilution from the signal.
The flow is data rather than a field on the model, so it travels through the
observation-data contract as `y_t = (y = concentrations, flow = flow)`.

```@example gs
fn = EpiSewer.FlowNormalize(EpiSewer.LogNormalError(; cv = Normal(0.1, 0.0)))
load = fill(2.0e13, 4)
daily_flow = [2.0e11, 4.0e11, 8.0e11, 2.0e11]
draw_fn = prior_draw(fn, load, (y = missing, flow = daily_flow))
(
    flow = daily_flow,
    expected_concentration = draw_fn.expected,
    drawn = round.(draw_fn.y_t; digits = 1),
)
```

Four times the flow on the same load gives a quarter of the concentration.

### `LogNormalError`

Relative measurement noise.
The real-space mean is the expected concentration and the real-space standard
deviation is proportional to it, so the parameter is a coefficient of variation
rather than an absolute scale.
That is the shape concentrations spanning orders of magnitude need.
The prior on it is sampled under the name `cv`, and its default is R's, which is
weak on purpose: a half-normal of scale 1, so a prior median coefficient of
variation of 0.67.
The data is expected to pull it far below that.

```@example gs
lne = EpiSewer.LogNormalError(; cv = Normal(0.3, 0.0))
map((100.0, 10000.0)) do level
    draws = prior_draw(lne, fill(level, 500)).y_t
    (
        level = level,
        sd = round(std(draws); sigdigits = 3),
        cv = round(std(draws) / mean(draws); digits = 2),
    )
end
```

A hundredfold rise in the expected concentration multiplies the standard
deviation by a hundred and leaves the coefficient of variation where it was.

### `LOD`

Left-censoring at a limit of detection.
A measurement at the limit scores the probability of being anywhere at or below
it, which is the convention EpiSewer's data uses: a non-detect is reported *at*
the limit rather than as a missing value.
Drawing from the prior with the limit above the expected concentration shows
most days coming back at the limit.

```@example gs
lod = EpiSewer.LOD(
    EpiSewer.LogNormalError(; cv = Normal(0.5, 0.0)); lod = 120.0
)
round.(prior_draw(lod, fill(100.0, 8)).y_t; digits = 1)
```

### `MeasurementOutliers`

Independent additive spikes on the expected series, drawn from a generalised
extreme value distribution truncated at zero.
The tail is extreme by design.
The median spike is a rounding error and the 99% quantile is seven orders of
magnitude larger, so most days are untouched and a rare day absorbs a large
excursion before the transmission dynamics have to explain it.

```@example gs
using Distributions: quantile

outliers = EpiSewer.MeasurementOutliers(
    EpiSewer.LogNormalError(; cv = Normal(0.1, 0.0)); scale = 100.0
)
spiked = prior_draw(outliers, fill(100.0, 200)).expected
(
    median_spike = round(quantile(outliers.spike, 0.5); sigdigits = 2),
    q99_spike = round(quantile(outliers.spike, 0.99); sigdigits = 2),
    days_within_1_percent = count(<=(101.0), spiked),
    q99_of_series = round(quantile(spiked, 0.99); sigdigits = 3),
)
```

### `DigitalPCRError`

Scores the positive partition counts directly rather than a concentration
derived from them.
The expected series it takes is the log expected copies per partition, and the
Poisson partition law turns that into the probability of a partition receiving
at least one copy.

```@example gs
dpcr = EpiSewer.DigitalPCRError(fill(20000, 5))
copies = [0.005, 0.01, 0.02, 0.05, 0.1]
draw_dpcr = prior_draw(dpcr, log.(copies))
(
    copies_per_partition = copies,
    positive_probability = round.(draw_dpcr.expected; digits = 4),
    positives = draw_dpcr.y_t,
)
```

Saturation comes for free, since the probability tends to one as the expected
copies grow.

## Customising `model()`

`model()`'s keywords come at two altitudes.
`infection_model` and `observation_model` replace a whole stage.
Everything else tunes one piece of the default stage, so `rt` changes the `R_t`
prior without touching the renewal process around it and the delay keywords
change one distribution without touching the chain around it.
Overriding one keyword leaves every other default in place, so a changed
assumption is a changed argument rather than a new model.

### The `R_t` latent process

`rt` takes any ecosystem latent model.
`RandomWalk` is the flexible non-parametric option, `AR` mean-reverts, and
`HilbertSpaceGP` is the basis-function approximation to a Gaussian process.
`CombineLatentModels` sums several of them and `TransformLatentModel` puts a
link on the result.

```@example gs
rw = EpiSewer.model(; assumptions..., rt = CT.RandomWalk())
(
    default = nameof(typeof(idm.infection_model.rt)),
    swapped = nameof(typeof(rw.infection_model.rt)),
)
```

The swapped model goes through `as_turing_model` exactly as the default does,
so the consequence of the swap can be read off its prior.

```@example gs
rw_draws = prior_draws(rw, (y = y, flow = flow), n)

series_plot(
    rt_prior, summarise(rw_draws, g -> exp.(g.Z_t), "random walk");
    ylabel = "R_t", xlabel = "day"
)
```

The random walk's interval is wider from the start and keeps widening as the
series runs on, because nothing pulls it back.
The Gaussian processes hold their length scale instead.

A `HilbertSpaceGP` measures its length scale in standard deviations of the time
index rather than in days, because it standardises the index.
A length scale taken from the literature in days therefore depends on the
series length, and `EpiSewer.gp_length_scale` does that conversion.
`n_gp` is the series length the default assumes, so a materially different
series needs either the keyword or a length scale converted by hand.

```@example gs
(
    days = 21.0,
    on_164_days = round(EpiSewer.gp_length_scale(21.0, 164); digits = 3),
    on_400_days = round(EpiSewer.gp_length_scale(21.0, 400); digits = 3),
)
```

### Stochastic infections and seeding

`infection_noise` takes the renewal modifier, and `nothing` removes it.
The renewal step type is where the choice shows up.

```@example gs
(
    stochastic = nameof(typeof(idm.infection_model.recurrent_step)),
    deterministic = nameof(
        typeof(
            EpiSewer.model(;
                assumptions..., infection_noise = nothing
            ).infection_model.recurrent_step
        )
    ),
)
```

`seeding` is a prior on log initial infections, and `initial_infections` is the
shorthand that centres a default `Normal` on it.
The scale matters, because the renewal process has to reconcile the seeded
infections with the measured concentrations, so a prior orders of magnitude
away from the data buys a sustained `R_t` excursion to compensate.
`EpiSewer.crude_initial_infections` reads a starting value off the data.
It averages the measurements in the first week, so it is computed on the series
actually observed: the shipped `initial_infections` comes from the thinned series
EpiSewer's README fits, and the full series gives a larger value.

```@example gs
crude = EpiSewer.crude_initial_infections(
    y, flow, assumptions.load_per_case
)
tuned = EpiSewer.model(; assumptions..., initial_infections = crude)
(
    crude = round(crude; digits = 1),
    seeding_prior = tuned.infection_model.initialisation,
)
```

### The infection model

`infection_model` replaces the renewal process outright, and it supersedes
`rt`, `infection_noise` and `seeding` because those exist to build the default
one.
Here a `DirectInfections` process replaces it, and the observation chain is
untouched.

```@example gs
direct = EpiSewer.model(;
    assumptions...,
    infection_model = CT.DirectInfections(;
        Z = CT.RandomWalk(), initialisation = Normal()
    )
)
(
    infection = nameof(typeof(direct.infection_model)),
    observation_unchanged = typeof(direct.observation_model) ===
        typeof(idm.observation_model),
    lead_in_unchanged = EpiSewer.observation_lead_in(direct) == lead_in,
)
```

The lead-in is a property of the observation chain alone, so swapping the
infection process never changes `n`.

### The observation error

`observation_model` replaces the whole chain, so a different error model means
rebuilding the chain around it.
The stages are the same ecosystem components the default uses, assembled
innermost first.

```@example gs
using Distributions: Gamma

function chain(error_model)
    obs = EpiSewer.FlowNormalize(error_model)
    obs = CT.LatentDelay(obs, assumptions.shedding_dist; D = 38.0)
    obs = CT.Ascertainment(
        obs, CT.FixedIntercept(log(assumptions.load_per_case))
    )
    obs = EpiSewer.LoadVariation(obs)
    return CT.LatentDelay(obs, assumptions.incubation_dist; D = 8.0)
end

censored = EpiSewer.model(;
    assumptions...,
    observation_model = chain(
        EpiSewer.LOD(EpiSewer.LogNormalError(); lod = 100.0)
    )
)
(
    stages = observation_chain(censored.observation_model),
    lead_in = EpiSewer.observation_lead_in(censored),
)
```

Same lead-in as the default, because the delays are the same and the error
model is not a delay.

### The observation-scale variance

Two keywords tune how much variation the chain can absorb before the
transmission dynamics have to account for it.
`load_cv` sets the individual-level load variation, and `outlier_scale` sets
the concentration equivalent of one unit of outlier spike.
`nothing` removes either component.

```@example gs
plain = EpiSewer.model(; assumptions..., outlier_scale = nothing)
(
    default = observation_chain(idm.observation_model),
    without_outliers = observation_chain(plain.observation_model),
)
```

Drawing from both shows what the outlier component buys.
Within a draw, most days sit close to the typical expected concentration, and
with the component in place a few carry a spike far above it.

```@example gs
using Statistics: median

function spread(m; n_draws = 50)
    ratios = DataFrame(
        ratio = reduce(
            vcat,
            [
                g.expected_y_t ./ median(g.expected_y_t)
                    for g in prior_draws(m, (y = y, flow = flow), n; n_draws)
            ]
        )
    )
    return (
        days = nrow(ratios),
        over_10x = nrow(@rsubset(ratios, :ratio > 10)),
        largest = round(maximum(ratios.ratio); sigdigits = 2),
    )
end

(with_outliers = spread(idm), without_outliers = spread(plain))
```

### The delay keywords

`generation_time`, `shedding_dist`, `incubation_dist` and `residence_dist` are
forwarded untouched to `Renewal` and `LatentDelay`, each of which dispatches on
what it is handed.
There are four forms.

A continuous `Distribution` is discretised by double-interval censoring, with
the matching `D_` keyword giving the right-truncation horizon.
An already discretised PMF vector is used verbatim, so the caller owns the
conventions, including no same-day transmission for a generation time.
A prior model such as an `UncertainDelay` carries priors on the distribution's
parameters, so the delay is inferred alongside everything else, and it holds
its own horizon.
`nothing` omits the convolution altogether.

`residence_dist` shows all four, because it is the one that defaults to
`nothing`: R's `residence_dist = c(1)` is a point mass at same-day arrival, so
the default chain has no residence wrapper.

```@example gs
delay_forms = (
    omitted = idm,
    continuous = EpiSewer.model(;
        assumptions..., residence_dist = Gamma(2.0, 1.0), D_residence = 5.0
    ),
    pmf = EpiSewer.model(; assumptions..., residence_dist = [0.5, 0.3, 0.2]),
    inferred = EpiSewer.model(;
        assumptions...,
        residence_dist = CT.UncertainDelay(
            Gamma, [Normal(2.0, 0.1), Normal(1.0, 0.1)]; D = 5.0, Δd = 1.0
        )
    ),
)
map(EpiSewer.observation_lead_in, delay_forms)
```

Each form moves the lead-in, and each moves it differently.
Adding a convolution lengthens the chain, so the infection series needs more
lead-in for the same observations.
A three-bin PMF costs two, a `Gamma` truncated at `D = 5.0` costs four, and the
`UncertainDelay` costs the same four because its horizon and bin width hold the
PMF length constant across draws.
Reading `n` back from `observation_lead_in` is what keeps a changed delay from
silently dropping observations.

## Learning more

- The [Model components](@ref model-components) page maps every EpiSewer
  component onto the ecosystem piece that provides it.
- The [Public API](@ref public-api) lists the full interface.
- To report a problem or ask a question, open an issue or a discussion on the
  [GitHub repository](https://github.com/seabbs/EpiSewer.jl).

The layout, navigation, and infrastructure of this site are generated by
[EpiAwarePackageTools](https://epiawarepackagetools.epiaware.org).
