# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Per-backend AD gradient test items. Each backend is its own `@testitem`,
# tagged so the per-backend CI can select it with a tag filter (e.g. `julia
# test/ad/runtests.jl enzyme_reverse`). Harness wiring lives in the managed
# `setup.jl`; the SCENARIOS come from the package's own `ADFixtures`
# registry.
#
# TOLERANCE: these items call the harness directly rather than through
# `setup.jl`'s `test_working_backend` shim, because they override `rtol` and the
# shim (being managed) does not forward it.
#
# The kit's default is `rtol = 5e-2`, far looser than any backend here needs.
# Measured 2026-08-17 against the ForwardDiff reference over all five scenarios,
# the worst relative deviation of any backend that answers correctly is 9.0e-16
# (Enzyme reverse; ReverseDiff both modes 5.8e-16, Mooncake reverse 7.4e-16,
# Mooncake forward 1.7e-16). At `5e-2` a wrong gradient in a small coordinate is
# invisible, because the comparison is norm-based: the weakest coordinate of
# each scenario needed a 257%-2336% error before it was noticed, so a real
# regression in one component could pass. `1e-6` leaves nine orders of magnitude
# of margin over the measured worst case — drift room for a dependency bump —
# while bounding the weakest coordinate to 0.005%-0.05%. Tightening it is what
# surfaced Enzyme forward's constant-offset gradient on the wastewater chain
# (see `ADFixtures.backend_broken_scenarios`).
#
# `atol` stays at the kit default: every scenario's reference norm is >= 15, so
# `rtol * norm` is >= 1.5e-5 and `atol` never binds.

@testitem "ForwardDiff gradients (marginal)" tags = [:ad, :forwarddiff] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "ForwardDiff";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "ReverseDiff (tape) gradients (marginal)" tags = [:ad, :reversediff] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "ReverseDiff (tape)";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "ReverseDiff (compiled) gradients (marginal)" tags = [:ad, :reversediff_compiled] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "ReverseDiff (compiled)";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "Enzyme forward gradients (marginal)" tags = [:ad, :enzyme, :enzyme_forward] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "Enzyme forward";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "Enzyme reverse gradients (marginal)" tags = [:ad, :enzyme, :enzyme_reverse] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "Enzyme reverse";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "Mooncake reverse gradients (marginal)" tags = [:ad, :mooncake, :mooncake_reverse] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "Mooncake reverse";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

@testitem "Mooncake forward gradients (marginal)" tags = [:ad, :mooncake, :mooncake_forward] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_working_backend(
        REG, "Mooncake forward";
        rtol = 1.0e-6, scenario_kwargs = (; category = :marginal)
    )
end

# Add latent (or other) scenario groups as the package needs, e.g.:
# @testitem "ForwardDiff gradients (latent)" tags = [:ad, :forwarddiff] setup = [ADHelpers] begin
#     EpiAwarePackageTools.test_working_backend(
#         REG, "ForwardDiff";
#         rtol = 1.0e-6, scenario_kwargs = (; category = :latent)
#     )
# end

# The registry's `category` selector is the only named option in this package
# with a fixed valid set, so it is the one place the kit's option-validation
# fuzzer applies. `scenarios` ignored an unrecognised category before, silently
# returning the marginal set, which would have let a mistyped category in an
# item above pass while testing the wrong scenarios (kit#310).
#
# Tagged `:forwarddiff` as well as `:registry` because `ad.yaml` selects items
# by backend tag alone, so a `:registry`-only item would never run in CI. The
# check is backend-independent; the ForwardDiff job is the reference job that
# always runs, so this executes once per CI run rather than seven times.
@testitem "scenario category is validated eagerly" tags = [:ad, :registry, :forwarddiff] setup = [ADHelpers] begin
    EpiAwarePackageTools.test_option_validation(
        ADFixtures.validate_category, ADFixtures.SCENARIO_CATEGORIES
    )
end
