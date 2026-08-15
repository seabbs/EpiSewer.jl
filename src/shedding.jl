module Shedding

# Shedding-module components for the EpiSewer port.
#
# These are `ComposableTuringIDModels`-compatible structs: each concrete
# component implements the corresponding `as_turing_model` method so it can be
# composed with the rest of the EpiAware ecosystem.

using Turing
using DynamicPPL
using Distributions
using ComposableTuringIDModels
import ComposableTuringIDModels: as_turing_model, as_turing_submodel

export LoadPerCase

"""
    LoadPerCase{T <: Distribution}

Calibrate the shed load per case (the `load_per_case_calibrate` component in
EpiSewer).

In EpiSewer the expected **load** arriving at the sampling site is the product
of the number of (effective) infections and the load shed per case:

```math
\\mathrm{load}_t = I_t \\times \\mathrm{load\\,per\\,case}.
```

`load_per_case` is an inferred (positive) parameter with its own prior, so its
posterior is informed by — and calibrates the absolute level of the estimated
infection series against — the observed case counts. This is the shedding-stage
mapping from the (latent / expected) infection series ``I_t`` to an expected
pathogen load.

`LoadPerCase` is a *modifier*: given an infection (or expected-infection) series
it returns the scaled expected-load series together with the sampled per-case
load parameter. It is deliberately role-light (subtypes the root
`AbstractComposableModel`) because it transforms a series rather than
constraining its own distribution, so it composes in among the infection /
shedding stages rather than as an observation-error model.

# Fields
- `load_per_case::T`: a positive prior (a bare `Distribution`, e.g.
  `LogNormal(0, 1)` or `HalfNormal(1)`), defaulting to `LogNormal(1, 1)`.

# Example
```julia
using EpiSewer, ComposableTuringIDModels
m = LoadPerCase()
model = as_turing_model(m, fill(100.0, 5))
```
"""
struct LoadPerCase{T <: Distribution} <: AbstractComposableModel
    load_per_case::T
end

LoadPerCase(; load_per_case::Distribution = LogNormal(1.0, 1.0)) = LoadPerCase(load_per_case)

"""
    as_turing_model(m::LoadPerCase, infections)

Sample the per-case load and return the scaled expected-load series.

`infections` is the infection (or expected-infection) series. The per-case load
is drawn once (a positive scalar prior) and the expected load is the
element-wise product `infections .* load_per_case`.

Returns the `(; expected_load, load_per_case)` NamedTuple, where
`expected_load` is the length-`length(infections)` scaled series and
`load_per_case` the sampled scalar.
"""
@model function as_turing_model(m::LoadPerCase, infections)
    # The per-case load: one positive scalar draw from its prior (a bare
    # `Distribution` draws a single scalar whatever shape is asked for).
    lpc ~ as_turing_submodel(m.load_per_case, 1)

    expected_load = infections .* lpc
    return (; expected_load = expected_load, load_per_case = lpc)
end

end # module Shedding
