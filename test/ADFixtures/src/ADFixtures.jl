# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Minimal AD-fixture registry implementing the EpiAwarePackageTools
# `ADRegistry` contract: scenarios (each with a ForwardDiff reference), a
# backend list, and broken/skip bookkeeping, consumed by the shared harness
# (`test/ad/setup.jl`). The scenarios are real log-densities exercising the
# package's own differentiable components.
#
# Each component's gradient scenario is a closed-form scalar log-density over
# the component's differentiable parameter(s) — the same mathematical core the
# component scores — so the per-backend AD matrix covers the new components.
# See https://github.com/EpiAware/EpiAwareADTools.jl.
module ADFixtures

using ADTypes: AutoForwardDiff, AutoReverseDiff, AutoMooncake,
    AutoMooncakeForward, AutoEnzyme
using DifferentiationInterface: DifferentiationInterface, Constant
import DifferentiationInterfaceTest as DIT
import ForwardDiff, ReverseDiff, Enzyme, Mooncake
using Distributions
using LogExpFunctions: logsumexp
using EpiSewer

export scenarios, backends, broken_scenario_names,
    backend_broken_scenarios, backend_skip_scenarios

# ForwardDiff reference gradient for a scenario function.
function _reference(f, θ, contexts)
    return DifferentiationInterface.gradient(
        f, AutoForwardDiff(), θ, contexts...
    )
end

"""
    scenarios(; with_reference = false, category = :marginal)

The AD gradient scenarios. Each is a `DIT.Scenario{:gradient, :out}` whose
`res1` carries a ForwardDiff reference when `with_reference = true`.

Every scenario is a closed-form differentiable log-density over a component's
differentiable parameter(s):

- `lod_censored_loglik` — the LOD left-censoring likelihood: a below-LOD term
  contributes `logcdf` and an above-LOD term the ordinary `logpdf`, as the
  [`EpiSewer.Measurements.LOD`](@ref) component scores, differentiated w.r.t.
  the observation-noise σ.
- `flow_normalize_loglik` — the flow-normalization likelihood: normalized
  observed and expected concentrations (scaled by `flow ./ reference_flow`)
  against a Normal error, differentiated w.r.t. the noise σ as the
  [`EpiSewer.Sewage.FlowNormalize`](@ref) component scores.
- `outlier_mixture_loglik` — the closed-form outlier-mixture log-density
  `log((1-p)·p_main + p·p_outlier)` that
  [`EpiSewer.Sampling.MeasurementOutliers`](@ref) scores per time point,
  differentiated w.r.t. the contamination probability `p`.
- `load_per_case_loglik` — the load-per-case log-density
- `digital_pcr_loglik` — the dPCR partition-count log-density (Binomial over cloglog-inverse-transformed expected copies)
  `logpdf(Normal(infections·lpc, σ), load)` that the expected load produced by
  [`EpiSewer.Shedding.LoadPerCase`](@ref) is scored against, differentiated
  w.r.t. the inferred per-case load `lpc`.
"""
function scenarios(; with_reference::Bool = false, category::Symbol = :marginal)
    out = DIT.Scenario{:gradient, :out}[]

    # --- LOD censored log-likelihood ---
    # θ = [σ]; expected Y=100, LOD=50; one censored obs (10, below LOD) and one
    # exact obs (120, above LOD).
    function lod_loglik(θ)
        σ = θ[1]
        y_cen = 10.0   # below LOD -> censored
        y_exact = 120.0
        return logcdf(Normal(100.0, σ), 50.0) + logpdf(Normal(100.0, σ), y_exact)
    end
    θ_lod = [10.0]
    push!(
        out,
        DIT.Scenario{:gradient, :out}(
            lod_loglik, θ_lod; name = "lod_censored_loglik",
            # ReverseDiff compiled traces the tape at the zero prep point;
            # σ = 0 is a degenerate Normal (-Inf), corrupting the tape.
            prep_args = (; x = [1.0], contexts = ()),
            res1 = with_reference ? _reference(lod_loglik, θ_lod, ()) : nothing
        )
    )

    # --- FlowNormalize log-likelihood ---
    # Observed concentrations normalized by flow/ref_flow, scored against
    # Normal(expected_norm, σ). θ = [σ].
    flow = [3.0e11, 2.0e11, 4.0e11]
    ref_flow = 3.0e11
    scale = [f / ref_flow for f in flow]
    y_obs = [150.0, 100.0, 200.0]       # raw observed concentrations
    Y_exp = [120.0, 120.0, 120.0]       # raw expected concentrations
    function flow_loglik(θ)
        σ = θ[1]
        y_norm = y_obs .* scale
        Y_norm = Y_exp .* scale
        return sum(logpdf.(Normal.(Y_norm, σ), y_norm))
    end
    θ_flow = [10.0]
    push!(
        out,
        DIT.Scenario{:gradient, :out}(
            flow_loglik, θ_flow; name = "flow_normalize_loglik",
            res1 = with_reference ? _reference(flow_loglik, θ_flow, ()) : nothing
        )
    )

    # --- MeasurementOutliers mixture log-likelihood ---
    # Closed-form two-component mixture over a single observed value. θ = [p].
    function outlier_loglik(θ)
        p = θ[1]
        y = 500.0
        main = Normal(100.0, 10.0)
        outlier = Normal(100.0, 50.0)   # wide component
        logp = log1p(-clamp(p, 0.0, 1.0)) + logpdf(main, y)
        logq = log(clamp(p, 1.0e-6, 1.0)) + logpdf(outlier, y)
        return logsumexp((logp, logq))
    end
    θ_out = [0.1]
    push!(
        out,
        DIT.Scenario{:gradient, :out}(
            outlier_loglik, θ_out; name = "outlier_mixture_loglik",
            # ReverseDiff compiled traces the tape at the zero prep point;
            # p = 0 sits on the mixture's clamp boundary, corrupting the tape.
            prep_args = (; x = [0.5], contexts = ()),
            res1 = with_reference ? _reference(outlier_loglik, θ_out, ()) : nothing
        )
    )

    # --- LoadPerCase log-likelihood ---
    # Scaled expected load infections·lpc scored against observed load with
    # Normal(error). θ = [lpc].
    infections = fill(100.0, 5)
    observed_load = [120.0, 90.0, 110.0, 95.0, 105.0]
    function lpc_loglik(θ)
        lpc = θ[1]
        # A scalar generator rather than a broadcast over
        # `infections .* lpc`, to keep no array temporary on the tape.
        return sum(
            logpdf(Normal(inf * lpc, 10.0), y)
                for (inf, y) in zip(infections, observed_load)
        )
    end
    θ_lpc = [1.0]
    push!(
        out,
        DIT.Scenario{:gradient, :out}(
            lpc_loglik, θ_lpc; name = "load_per_case_loglik",
            res1 = with_reference ? _reference(lpc_loglik, θ_lpc, ()) : nothing
        )
    )

    # --- DigitalPCR (dPCR partition-count) log-likelihood ---
    # Observed positive partition counts scored against Binomial(total, p)
    # with p = 1 - exp(-exp(Y_t)) (cloglog-inverse transform of the
    # log-copies-per-partition expectation). θ = [log_copies] per time point.
    total_parts = [1000, 1000, 1000]
    positives = [10, 25, 40]
    function dpcr_loglik(θ)
        # Per-time-point log expected copies per partition -> cloglog-inverse
        # -> Binomial success probability. A generator (not a fused broadcast)
        # keeps `n` an Int under ReverseDiff's broadcast machinery, which
        # would otherwise promote it to a Dual and fail `Int(dual)`.
        p_t = clamp.(1.0 .- exp.(-exp.(θ)), 1.0e-6, 1 - 1.0e-6)
        return sum(
            logpdf(Binomial(n, p), y) for (n, p, y) in zip(total_parts, p_t, positives)
        )
    end
    θ_dpcr = log.([0.01, 0.025, 0.04])
    push!(
        out,
        DIT.Scenario{:gradient, :out}(
            dpcr_loglik, θ_dpcr; name = "digital_pcr_loglik",
            res1 = with_reference ? _reference(dpcr_loglik, θ_dpcr, ()) : nothing
        )
    )

    return out
end

"""
    backends()

The AD backends to test, as `(; name, backend)` named tuples. Seeded to match
every backend `test/ad/scenarios.jl` emits a testitem for, so a fresh package
passes its AD suite out of the box; trim to the subset the package actually
supports.
"""
function backends()
    return [
        (name = "ForwardDiff", backend = AutoForwardDiff()),
        (name = "ReverseDiff (tape)", backend = AutoReverseDiff(compile = false)),
        (name = "ReverseDiff (compiled)", backend = AutoReverseDiff(compile = true)),
        # `function_annotation = Enzyme.Const`: the scenario closures capture the
        # fixture data but carry no derivative data of their own. Without it
        # Enzyme raises `EnzymeMutabilityException` ("argument cannot be proven
        # readonly") on any closure that broadcasts over a captured array.
        (
            name = "Enzyme forward",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Forward),
                function_annotation = Enzyme.Const
            ),
        ),
        (
            name = "Enzyme reverse",
            backend = AutoEnzyme(
                mode = Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation = Enzyme.Const
            ),
        ),
        (name = "Mooncake reverse", backend = AutoMooncake(config = nothing)),
        (name = "Mooncake forward", backend = AutoMooncakeForward()),
    ]
end

"Scenario names broken on every backend."
broken_scenario_names() = String[]

"Per-backend broken scenario names (`Dict{String, Set{String}}`)."
backend_broken_scenarios() = Dict{String, Set{String}}()

"Per-backend scenario names too unstable to run at all."
backend_skip_scenarios() = Dict{String, Set{String}}()

end # module ADFixtures
