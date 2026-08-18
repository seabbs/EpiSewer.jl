# README wording checks: the kit's opt-in helpers, which the managed
# `test/package/quality.jl` does not call. `test_readme_sections` (structure)
# runs there already; these three check what the sections say.
#
# Tagged `:readme`, the tag `test/runtests.jl` already routes with its
# `readme_only` filter. Untagged `:quality`, so `skip_quality` keeps them: they
# read one file and need no subprocess.

@testitem "README bullets carry the why" tags = [:readme] begin
    using EpiAwarePackageTools: test_readme_bullets
    # The "Why EpiSewer.jl?" section, checked for 3-6 bullets. The ecosystem
    # convention is that this section is bullets and nothing else, so an
    # explanatory paragraph creeping back in shows up as a bullet count.
    test_readme_bullets(joinpath(@__DIR__, ".."))
end

@testitem "README has no seeded placeholders left" tags = [:readme] begin
    using EpiAwarePackageTools: test_readme_placeholders
    test_readme_placeholders(joinpath(@__DIR__, ".."))
end

@testitem "README reads as plain prose" tags = [:readme] begin
    using EpiAwarePackageTools: test_readme_prose
    # Banned words and a 40-word sentence cap. Nothing is filtered from the
    # default banned list: none of the words on it is a term of art here.
    test_readme_prose(joinpath(@__DIR__, ".."))
end
