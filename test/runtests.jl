# PACKAGE-OWNED — scaffold writes this once and never overwrites it.
#
# Main test entry. Discovers `@testitem`s (the managed QA testset under
# `test/package/` plus the package's own unit tests) with TestItemRunner.
# `:ad`-tagged items live under `test/ad/` with their own environment and
# run in dedicated per-backend CI, so are excluded here (test/ad/runtests.jl).
#
# Filters:
#   skip_quality  — skip the QA testset (fast local iteration)
#   quality_only  — run only the QA testset
#   readme_only   — run only `:readme`-tagged items (README/tutorial tests)

using TestItemRunner: run_tests

# `run_tests` is TestItemRunner's public entry point and roots discovery at the
# directory it is given, so passing `@__DIR__` scopes the scan to this package's
# own `test/` tree: a nested worktree under the repo root (`worktrees/wt-*`) is
# never walked and cannot inject test items or shadow a same-named
# `@testsnippet`.
#
# This deliberately does NOT use `EpiAwarePackageTools.run_package_tests`, which
# reimplements discovery in order to derive the package name from the parent
# directory and so provide an implicit `using EpiSewer` in every item. That
# reimplementation calls TestItemRunner internals (`run_testitem`,
# `TestItemDetection`, `JuliaSyntax`), and TestItemRunner 1.2.1 changed
# `run_testitem`'s signature, which failed every item in the suite on every OS.
# Every `@testitem` here imports what it needs explicitly, so the implicit
# import buys nothing and the public API is enough.
# See EpiAware/EpiAwarePackageTools.jl#451.

if "skip_quality" in ARGS
    run_tests(
        @__DIR__;
        filter = ti -> !(:quality in ti.tags) && !(:ad in ti.tags)
    )
elseif "quality_only" in ARGS
    run_tests(@__DIR__; filter = ti -> :quality in ti.tags)
elseif "readme_only" in ARGS
    run_tests(@__DIR__; filter = ti -> :readme in ti.tags)
else
    run_tests(@__DIR__; filter = ti -> !(:ad in ti.tags))
end
