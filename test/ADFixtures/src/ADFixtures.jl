# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# AD-fixture registry implementing the EpiAwarePackageTools `ADRegistry`
# contract. The scenarios are REAL differentiable log-densities from the
# package: the (linked) log-joint of composed `IDModel`s and observation models
# built from EpiSewer's own components and conditioned on data simulated from
# their own prior — the gradients an AD backend must get right for NUTS to work.
# Each scenario carries a ForwardDiff reference gradient. The shared harness
# (driven from `test/ad/setup.jl`) consumes this registry.
module ADFixtures

using ADTypes: AutoForwardDiff
using DifferentiationInterface: DifferentiationInterface
import DifferentiationInterfaceTest as DIT
import ForwardDiff
using ComposableTuringIDModels: Ascertainment, DirectInfections, HalfNormal,
    IDModel, LatentDelay, NormalError, RandomWalk, Renewal, as_turing_model
using Distributions
using EpiSewer: EpiSewer
using Random: Random, MersenneTwister
using Statistics: median
using DynamicPPL: DynamicPPL, LogDensityFunction, VarInfo, getindex_internal,
    link, getlogjoint
import LogDensityProblems as LDP

export scenarios, backends, broken_scenario_names,
    backend_broken_scenarios, backend_skip_scenarios,
    SCENARIO_CATEGORIES, validate_category

# Turn a DynamicPPL model into a real differentiable scalar log-density.
#
# We link the model's `VarInfo` so every constrained variable (positive standard
# deviations, GEV-supported outlier spikes, ...) maps to an unconstrained real
# coordinate. The returned `f(θ)` is then the log-joint (including the linking
# log-Jacobian) over all of ``ℝ^d`` — exactly the target a gradient-based
# sampler differentiates.
function _logdensity(model)
    vi = link(VarInfo(model), model)
    ldf = LogDensityFunction(model, getlogjoint, vi)
    return θ -> LDP.logdensity(ldf, θ), LDP.dimension(ldf)
end

# One prior realisation of `m`: the simulated observations, and the linked
# parameter vector that produced them.
#
# Each scenario conditions on those observations and takes its gradient at that
# same vector, so the target sits at the draw that generated the data and the
# residuals sit at the noise scale the model itself drew. At an arbitrary
# starting point (`0.3 .* randn`, say) the observations are an order of
# magnitude away from the expected series instead, and then a single coordinate
# — the noise scale — carries almost the whole gradient norm, which hides every
# other coordinate from the harness's norm-based `isapprox` comparison.
#
# `y_free` is the unobserved data argument to simulate under, and `ymap` shapes
# the drawn vector into the argument the conditioned model takes.
function _prior_draw(m, shape, seed::Int; y_free = missing, ymap = identity)
    free = as_turing_model(m, y_free, shape)
    vi = VarInfo(MersenneTwister(seed), free)
    is_obs = k -> startswith(string(k), "y_t")
    ks = collect(keys(vi))
    y = reduce(vcat, [getindex_internal(vi, k) for k in filter(is_obs, ks)])
    linked = link(vi, free)

    model = as_turing_model(m, ymap(y), shape)
    f, dim = _logdensity(model)
    # The conditioned model's own coordinate order, filled from the draw above.
    vi_cond = link(VarInfo(model), model)
    θ = reduce(vcat, [getindex_internal(linked, k) for k in keys(vi_cond)])
    @assert length(θ) == dim "simulated point does not fill the model's coordinates"
    return model, f, θ
end

# Short shared fixtures: the suite is a gradient check, not a fit, so eight time
# points and three-lag PMFs are enough to exercise every scan and convolution.
# Concentrations stay on EpiSewer's gc/mL scale and flows on its mL/day scale so
# the gradients see the magnitudes the real model sees.
const _N = 8
const _GEN_INT = [0.2, 0.3, 0.5]
const _SHED_PMF = [0.5, 0.3, 0.2]
const _FLOW = [2.5e11, 3.0e11, 2.8e11, 3.4e11, 2.6e11, 3.1e11, 2.9e11, 3.3e11]
# Load shed per case (gc/case), EpiSewer's prior median for the Zurich example.
const _LPC = 2.0e11
# Concentration equivalent (gc/mL) of one unit of outlier spike, built as the R
# package builds it: `log(load_mean) + log(ε_t) - flow_median_log` with
# `load_mean` the load shed per case (`EpiSewer_main.stan:760`, the additive
# outlier component). So one unit of spike is one case-equivalent of load spread
# over a day's flow.
const _OUTLIER_SCALE = _LPC / median(_FLOW)

# The infection model the concentration-measurement scenarios share: a random
# walk on log concentration around EpiSewer's scale, so each scenario isolates
# one measurement component.
_direct() = DirectInfections(;
    Z = RandomWalk(), initialisation = Normal(log(100.0), 0.2)
)

# Build the registry's targets once, as `(name, f, θ, prep)` tuples: each is a
# posterior conditioned on data simulated from its own prior with a fixed
# per-scenario seed, differentiated at the parameters that produced that data
# (see [`_prior_draw`](@ref)), so every backend process sees the same target.
# `prep` overrides DIT's tape-preparation point, `nothing` keeps its default.
function _targets()
    out = Tuple{String, Any, Vector{Float64}, Union{Nothing, Vector{Float64}}}[]

    # --- LOD left-censoring -------------------------------------------------
    lod = 50.0
    lod_model = IDModel(
        _direct(), EpiSewer.LOD(NormalError(; std = HalfNormal(10.0)); lod = lod)
    )
    # A measurement reported at the LOD scores `logcdf`, so the censored branch
    # of the likelihood is on the gradient path and not only the exact one.
    censor_one = y -> (yc = copy(y); yc[3] = lod; yc)
    _, f_lod, θ_lod = _prior_draw(lod_model, _N, 101; ymap = censor_one)
    # DIT prepares at `zero(x)` by default, and this is the one scenario that
    # cannot be traced there: at the origin the expected concentration is 100
    # gc/mL with σ = 1, so the censored measurement at the 50 gc/mL limit sits
    # ~50σ into the left tail (log-density -2.3e5). A compiled ReverseDiff tape
    # traced there returns `-Inf` for the primal at every later point, failing 12
    # of the backend's assertions. Preparing a short step away from the
    # evaluation point keeps DIT's prep-reuse check meaningful while staying in
    # the region where the censoring term is well conditioned.
    push!(out, ("LOD censored posterior", f_lod, θ_lod, θ_lod .+ 0.1))

    # --- LogNormalError -----------------------------------------------------
    # The relative-noise family the wastewater model uses by default: its σ is a
    # coefficient of variation, so the gradient runs through `reparameterise`.
    lognormal_model = IDModel(_direct(), EpiSewer.LogNormalError())
    _, f_ln, θ_ln = _prior_draw(lognormal_model, _N, 102)
    push!(out, ("LogNormalError posterior", f_ln, θ_ln, nothing))

    # --- MeasurementOutliers ------------------------------------------------
    # Integrated outlier detection: an i.i.d. GEV spike per day added to the
    # expected series. `scale` puts the spike in gc/mL (see `_OUTLIER_SCALE`),
    # so the additive term is on the same footing as the measurement noise — a
    # spike of one case-equivalent shifts the expected concentration by ~0.7
    # gc/mL against a drawn noise σ of the same order. `NormalError` is what
    # puts σ on the gc/mL scale here; the default chain's `LogNormalError` is
    # CV-parameterised, so its σ is a ratio and does not scale with the signal.
    outlier_model = IDModel(
        _direct(),
        EpiSewer.MeasurementOutliers(
            NormalError(; std = HalfNormal(10.0)); scale = _OUTLIER_SCALE
        )
    )
    _, f_out, θ_out = _prior_draw(outlier_model, _N, 103)
    push!(out, ("MeasurementOutliers posterior", f_out, θ_out, nothing))

    # --- DigitalPCRError ----------------------------------------------------
    # The dPCR partition likelihood takes log copies per partition as its
    # expected series, so an infection model cannot feed it directly. Wrapping
    # it in an `Ascertainment` whose latent is a `RandomWalk` added to a fixed
    # base series is the meaningful way to reach its gradient (as CTIDM reaches
    # `BinomialError`'s).
    dpcr_model = Ascertainment(
        EpiSewer.DigitalPCRError(fill(1_000, _N)), RandomWalk();
        transform = (Y_t, x) -> Y_t .+ x, latent_prefix = ""
    )
    dpcr_base = fill(log(0.02), _N)
    # Partition counts are counts: a `VarInfo` stores the drawn values as
    # floats, and `BinomialError` needs them back as `Int`.
    _, f_dpcr, θ_dpcr = _prior_draw(
        dpcr_model, dpcr_base, 104; ymap = y -> round.(Int, y)
    )
    push!(out, ("DigitalPCRError ascertainment posterior", f_dpcr, θ_dpcr, nothing))

    # --- the full wastewater chain ------------------------------------------
    # The chain `EpiSewer.model()` assembles, at AD scale: renewal infections,
    # per-case load scaling, shedding delay, flow division, log-normal noise.
    # This covers `FlowNormalize` and the composition, so no separate
    # flow-normalisation scenario is needed.
    full_model = IDModel(
        Renewal(;
            generation_time = _GEN_INT, rt = RandomWalk(),
            initialisation = Normal(log(50.0), 0.2)
        ),
        Ascertainment(
            LatentDelay(
                EpiSewer.FlowNormalize(EpiSewer.LogNormalError()), _SHED_PMF
            ),
            Normal(log(_LPC), 0.5)
        )
    )
    # The flow is data on both sides of the draw.
    _, f_full, θ_full = _prior_draw(
        full_model, _N, 105;
        y_free = (y = missing, flow = _FLOW), ymap = y -> (y = y, flow = _FLOW)
    )
    push!(out, ("Wastewater chain posterior", f_full, θ_full, nothing))

    return out
end

"""
    SCENARIO_CATEGORIES

The group names [`scenarios`](@ref) accepts. One entry today: every scenario in
this registry differentiates a marginal log-density.
"""
const SCENARIO_CATEGORIES = (:marginal,)

"""
    validate_category(category)

Reject a `category` outside [`SCENARIO_CATEGORIES`](@ref).

Eager, because the selector is otherwise unused: an unrecognised name would
return the whole registry and the typo would surface as a passing test of the
wrong scenarios.
"""
function validate_category(category::Symbol)
    category in SCENARIO_CATEGORIES && return nothing
    return throw(
        ArgumentError(
            "unknown scenario category $(repr(category)); valid categories: " *
                join(repr.(SCENARIO_CATEGORIES), ", ")
        )
    )
end

@doc """
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios — each a `DIT.Scenario{:gradient, :out}` over a real
EpiSewer log-density: the linked log-joint of a composed model conditioned on
data simulated from its own prior, differentiated at the parameters that
produced that data. When `with_reference = true` each scenario carries its
ForwardDiff reference gradient in `res1`. `category` is the harness's group
selector, validated against [`SCENARIO_CATEGORIES`](@ref): every scenario here
is in the single `:marginal` group, so a mistyped category throws rather than
silently returning the marginal set under another name.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    validate_category(category)
    out = DIT.Scenario{:gradient, :out}[]
    for (name, f, θ, prep) in _targets()
        ref = with_reference ?
            DifferentiationInterface.gradient(f, AutoForwardDiff(), θ) :
            nothing
        prep_kwargs = prep === nothing ? (;) :
            (; prep_args = (; x = prep, contexts = ()))
        push!(
            out,
            DIT.Scenario{:gradient, :out}(
                f, θ; name = name, res1 = ref, prep_kwargs...
            )
        )
    end
    return out
end

@doc """
    backends()

The AD backends exercised against the scenarios, as `(; name, backend)` named
tuples: ForwardDiff (the reference), ReverseDiff (tape and compiled), Mooncake
reverse and forward, and Enzyme reverse and forward — the full seven-backend
matrix `ad.yaml` runs in CI. Failures are recorded in
[`backend_broken_scenarios`](@ref) rather than by trimming this list.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = _forwarddiff()),
        (name = "ReverseDiff (tape)", backend = _reversediff()),
        (name = "ReverseDiff (compiled)", backend = _reversediff_compiled()),
        (name = "Enzyme forward", backend = _enzyme_forward()),
        (name = "Enzyme reverse", backend = _enzyme()),
        (name = "Mooncake reverse", backend = _mooncake()),
        (name = "Mooncake forward", backend = _mooncake_forward()),
    ]
end

# Backend constructors are written so that loading a backend package is only
# required when that backend is actually requested (the AD env loads them all,
# but this keeps the registry importable without every backend present).
_forwarddiff() = AutoForwardDiff()
_adtypes() = _require("47edcb42-4c32-4615-8424-f2b9edc5f35b", "ADTypes")
# Load the backend package, not just `ADTypes`. `AutoReverseDiff` and
# `AutoMooncake` are only descriptions; DifferentiationInterface dispatches on
# them through package extensions, which activate when the backend package
# itself is loaded. Constructing the type without loading the package yields a
# backend that fails at the first `gradient` call.
#
# `test/ad/setup.jl` loads all four explicitly, so the AD suite never saw this.
# The docs' AD comparison page does not, and silently lost both ReverseDiff
# modes and both Mooncake modes from its table — Enzyme survived only because
# its constructor below already required its package.
function _require(uuid, name)
    return Base.require(Base.PkgId(Base.UUID(uuid), name))
end
function _reversediff_auto(compile::Bool)
    _require("37e2e3b7-166d-5795-8a7a-e32c996b4267", "ReverseDiff")
    return _adtypes().AutoReverseDiff(; compile = compile)
end
_reversediff() = _reversediff_auto(false)
_reversediff_compiled() = _reversediff_auto(true)
function _mooncake()
    _require("da2b9cff-9c12-43a0-ae48-6db2b0edb7d6", "Mooncake")
    return _adtypes().AutoMooncake(; config = nothing)
end
function _mooncake_forward()
    _require("da2b9cff-9c12-43a0-ae48-6db2b0edb7d6", "Mooncake")
    return _adtypes().AutoMooncakeForward(; config = nothing)
end
function _enzyme()
    Enzyme = Base.require(
        Base.PkgId(
            Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9"), "Enzyme"
        )
    )
    # `function_annotation = Enzyme.Const`: the log-density closures carry no
    # derivative data, and without it Enzyme raises `EnzymeMutabilityException`
    # ("argument cannot be proven readonly") on every DynamicPPL log-density.
    return _adtypes().AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation = Enzyme.Const
    )
end
function _enzyme_forward()
    Enzyme = Base.require(
        Base.PkgId(
            Base.UUID("7da242da-08ed-463a-9acd-ee780be4f1d9"), "Enzyme"
        )
    )
    # Same `function_annotation = Enzyme.Const` rationale as `_enzyme()`,
    # forward mode.
    return _adtypes().AutoEnzyme(;
        mode = Enzyme.set_runtime_activity(Enzyme.Forward),
        function_annotation = Enzyme.Const
    )
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

@doc """
    backend_broken_scenarios()

Per-backend broken scenario names (`Dict{String, Set{String}}`).

The harness's `check_broken` computes the pass/fail boolean itself and only
falls back to `@test_broken` when a listed scenario actually fails, so
over-listing is safe and under-listing reds CI. Listing every scenario for a
backend is not safe in a different sense: it empties the set the full
correctness sweep runs over, so that backend stops being tested.

Add an entry only with a measured failure.
"""
function backend_broken_scenarios()
    # Measured 2026-08-17. Enzyme forward returns a finite but wrong gradient
    # for the full chain: ‖g - ref‖ = 204 against a reference norm of 86.7, a
    # relative deviation of 2.35. The error is a pure constant offset — every
    # one of the 12 coordinates is high by 58.93124348067019, with a spread of
    # 9.4e-15 across them — which is the same shape ComposableTuringIDModels'
    # registry records for Enzyme forward on its `CombineInfections+Split`
    # scenario, so it is likely one upstream fault rather than two.
    #
    # This is a backend fault, not a model one: on the same scenario Enzyme
    # reverse agrees with the reference to 9.0e-16, both ReverseDiff modes to
    # 5.8e-16, Mooncake reverse to 7.4e-16 and Mooncake forward to 1.2e-16, and
    # Enzyme forward itself agrees to <= 3.8e-16 on the other four scenarios.
    # Only this one pairing fails.
    return Dict{String, Set{String}}(
        "Enzyme forward" => Set(["Wastewater chain posterior"]),
    )
end

"Per-backend scenario names too unstable to even run (segfault/hang)."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
