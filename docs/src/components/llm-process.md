# LLM-Assisted Development Process

## Aim

This package is an experiment to see if non-frontier models can be used to effectively port R packages to Julia when using composable elements built by frontier models.

## Methodology

- **Code harness**: pi coding agent with subagents (pi-subagents)
- **Language model**: DeepSeek V4 Flash via pi-agent (openrouter/~deepseek/deepseek-v4-flash-latest)
- **Date**: August 2026
- **Cost**: $13.62 on OpenRouter, part of an $18.82 total across both non-frontier passes.

## Review pass

- **Date**: 2026-08-15. **Harness**: pi coding agent with pi-subagents (same as the implementation pass).
- **Review model**: Kimi 3 as the overseer, delegating to DeepSeek V4 Flash subagents (openrouter/~deepseek/deepseek-v4-flash-latest) as the weak workers. Both non-frontier, so the experiment covers non-frontier review of non-frontier work.
- **Cost**: $5.20 on OpenRouter, the remainder of the $18.82 total.
- **Instructions**: review each section of the replication prompt against the repository state; get CI green (quality suite, JET, AD, Documenter); correct drift from the EpiAwarePackageTools README standard; make the README worked example runnable code for the Documenter pipeline rather than saved plots and scripts; align the example data with the original EpiSewer example (sparse Monday/Thursday measurements over the same window); enforce ecosystem reuse over custom code; keep code and comments concise and elegant; work through heavy weak-worker subagent delegation with the review model as overseer, committing and pushing each fix.

**Operator bumps**: the implementation agent stopped work twice and had to be bumped by the operator — (1) after component development, leaving empty placeholder modules, an outdated model-components table, and the examples section unstarted (operator sent a "keep-going" prompt); (2) at the final checkpoint, stopping with CI still red and asking whether to continue.

## Frontier-model review pass

The experiment's premise is that a non-frontier model can port a package when the composable pieces are supplied by frontier models.
Testing that premise needs a frontier model to check the result, so a third pass audited the two non-frontier passes above.

- **Date**: 2026-08-17.
- **Harness**: Claude Code 2.1.233 (CLI), main agent plus `Task` subagents.
- **Review model**: Claude Opus 5 (1M-token context) for the main agent and every subagent.
- **Scale**: one main agent plus 11 subagents; ~3,200 model calls, ~1.4 million output tokens and ~834 million input-side tokens (almost all of it prompt-cache reads), measured from the session transcripts.
- **Cost**: $Z (placeholder — to be filled in once the project is complete).
- **Instructions**: work through every bullet of the replication prompt and record its true status against the repository rather than against the previous passes' own session notes; fix the failing Enzyme AD CI jobs; and judge how closely the package tracks `ComposableTuringIDModels.jl` in style and in reuse.
- **Output**: 24 commits, 10 issues filed on this repository and 5 upstream across three ecosystem packages.

The scale ratio is the part worth recording.
Auditing the port cost roughly two orders of magnitude more inference than producing it: the two non-frontier passes together came to $18.82, and this pass consumed ~834 million input-side tokens against their output.
That is not a like-for-like comparison — reading a codebase to check it is inherently heavier than writing it, and prompt-cache reads are cheap per token — but it does undercut a tempting reading of the experiment.
If a cheap model ports a package and an expensive one is needed to find out whether the port is right, the saving is smaller than the model prices suggest.

### What the review found

Most bullets were delivered, and the ecosystem carried the port.
Five new structs (`LOD`, `LogNormalError`, `DigitalPCRError`, `MeasurementOutliers`, `FlowNormalize`) cover a model whose R original is assembled from about twenty components; everything else is composition of existing `ComposableTuringIDModels.jl` pieces.
The package built, its tests passed, its documentation read plausibly, and it produced a fitted `R_t` in a believable range.

It was also wrong in four ways that mattered, none of which any test caught.

1. **The generation interval was indexed one lag too early.**
   Its values matched R's `get_discrete_gamma_shifted` to four decimal places, but R's formula runs `k <- 1:maxX`, so R's first bin is lag 1 while ours was lag 0.
   The model therefore allowed 28.7% of transmission to happen on the day of infection, and its mean generation interval was 1.96 days rather than the 2.94 the parameterisation asks for.
   A generation interval a day too short compresses estimated `R_t` toward 1.
2. **The incubation-period convolution was missing entirely.**
   It was computed and never used.
   Because the shedding load is indexed from symptom onset, omitting it applied the whole shedding profile about 3.4 days early.
   The README explains why the incubation period is needed, and then the model did not use it.
3. **Thirty-one per cent of the observations were never scored.**
   Each delay convolution shortens the expected series, and the observation loop aligns to its end, so the first 37 of 120 measurements contributed nothing to the likelihood.
   Setting the first observation to an absurd value left the log-joint bit-identical.
   The returned series was still full length, which is what hid it.
4. **The worked example fitted the wrong dataset.**
   EpiSewer's README example deliberately thins its measurements to Mondays and Thursdays and then shows the fit recovering the withheld days.
   Ours fitted 117 of 120 days, so the missing-data behaviour the example exists to demonstrate was exercised on three days instead of eighty-six.

The pattern is consistent, and it is the useful result of the experiment.
Every one of these is a composition error or an off-by-one in an index, not a mistake inside a component.
Each component was individually correct and individually tested.
The errors live in how the pieces were joined and in what the numbers meant, which is exactly what a unit test of a component cannot see.

Two further findings concern how the work was verified rather than what it produced.

**The test suite included a whole category of tests that could not fail.**
Every component had an automatic-differentiation scenario, and no scenario called the component: each re-implemented the component's log-density inside the fixture file.
A seven-backend AD matrix was reporting green over code the package never executed.
Rewiring those scenarios to differentiate the real components found a genuine bug within minutes — a `hasmethod` check reachable on the differentiated path, which Mooncake has no rule for — that the previous scenarios could not have detected, because the closure they differentiated contained none of the code at fault.

The same pattern held in the unit tests, and there it was countable.
Seven test items asserted only that the code had run: `mdl !== nothing`, or that the sampler returned the number of draws it was asked for.
One of them was self-documenting.
`test/sewage.jl` asserted `mdl !== nothing` with a comment saying the flow division's correctness was "covered by the integration tests", and the integration test it pointed at asserted `mdl !== nothing`.
Two tests each deferring to the other, so flow normalisation — the package's central transformation — had no assertion anywhere in the suite.
That is not a gap someone forgot to fill; it is a gap that documented itself as filled.

Clearing all seven took the suite from 194 assertions to 192, because the replacements were more precise rather than more numerous.

Two of the six test files contained no such tests at all, and the reason generalises past this project.
`test/data.jl` and `test/distributions.jl` check specific counts, dates, types, and numbers computed from the R original — the mean generation interval against a transcription of R's own discretiser, the shedding mode, the window boundaries.
An assertion whose expected value has to come from outside the code under test cannot be vacuous.
In every instance found here, the smoke was in a test whose expected value came from the model itself, or from nothing at all.

**Infrastructure consumed the most effort and produced the longest-lived failure.**
The Enzyme jobs stayed red for two days over a missing `function_annotation = Enzyme.Const` on two backend constructors, a one-line annotation that `ComposableTuringIDModels.jl` already carries with the reason written beside it.

### What the review had to change

The audit was not read-only.
Nine defects were fixed in the same pass, in 24 commits.

| Defect | Effect before the fix |
|---|---|
| Generation interval indexed one lag early | mean 1.96 days instead of 2.94, and 28.7% of transmission on the day of infection |
| Incubation convolution absent | the whole shedding profile applied ~3.4 days early |
| Lead-in not accounted for in `n` | 44 of 120 observations never scored |
| Worked example on the dense series | the missing-data demonstration exercised on 3 days instead of 86 |
| `MeasurementOutliers` a contamination mixture | a different model from `outliers_estimate` |
| Spike prior untruncated | 36.8% of prior mass on negative spikes Stan forbids |
| `DigitalPCRError` data path | a `missing` count frozen at the first draw and scored as data |
| `hasmethod` on the differentiated path | Mooncake reverse failed; other backends silently tolerated it |
| Enzyme backends missing `function_annotation` | two AD jobs red for two days |

Alongside those: four unused dependencies removed from the package (one of them pulling the whole Makie stack), the AD gradient tolerance tightened from the kit's default by five orders of magnitude, seven vacuous test items cleared, nine dead documentation links fixed, and five table rows corrected that described designs the code had replaced.

Two things could not be fixed from here and are recorded as issues: the `Auto Version Increment` workflow needs a repository setting only a maintainer can change, and the coverage upload needs a token.
One more is left open deliberately — the worked example's plots have not been regenerated, because the fit they came from used a 60-draw fallback on the wrong data, and replacing them with a fit nobody has verified would repeat the mistake being corrected.

### What the review got wrong

Two of the gaps this pass reported upstream as missing ecosystem capabilities were not missing.
Stochastic infection noise and an estimated seeding phase are both reachable through documented, `public` extension points — a renewal modifier's returned incidence re-enters the recursion, and the initialisation window can be supplied by a custom step.
Both issues were retracted after the operator questioned the first one.
The lesson generalises past this project: an audit that concludes a capability is absent should trace the extension seams before saying so, and a reviewer is as capable of a confident wrong answer as the work under review.

The full bullet-by-bullet audit is kept with the other review notes in the gitignored `.resources/` directory.

## Source material

The original EpiSewer R package source code was embedded as a resource at `.resources/EpiSewer/`.

## The replication prompt

The prompt that guided this project is reproduced in full below, as prose rather than as a code block.
It lives on this page because the original `.resources/prompt.md` is not committed (`.resources` is gitignored).
Only three things have changed: the heading levels are shifted to sit under this section, one unbalanced inline-code marker is closed, and the cost placeholder's dollar sign is escaped so it is not read as maths.
The wording, ordering and typographical errors are exactly as the implementing agents received them.

### Replication plan for EpiSewer

#### Aim

I want to replicate the model functionality but not interface or other tools (i.e. plotting, docker etc) of the EpiSewer package using tools from the EpiAware.org GitHub Ecosystem

#### Setup

- We want to first make sure we can install and run `EpiSewer`
- We will first set the Package up using `EpiAwarePackageTools.jl`.
- Add `.resources` to the `.gitignore`
- Our core tools and dependencies will be `ComposableTuringIDModels.jl`, `CensoredDistributions.jl`, `EpiAwareADTools.jl`. This is on top of the tools package tools will bring.
- We know there are issues declaring as not part of the EpiAware.org ecosystem so we may need to make a few tweaks so that the docs show up in our samabbott.co.uk/EpiSewer.jl.
- We need to opt in to AD benchmarking
- We will want to flesh out what this package is i.e its readme folling EpiAware standards
- We need to add a clear derived from EpiSewer attribution
- We will want to make a documentation page that details the LLM process used to create the package. It should have this document in it as i.e. the prompt. It should start with the aim (experimenting to see if non-frontier models can be used to effectively language port packages when using composable elements built by frontier models). You should record in this document the code harness we are using as well the as the LLM model used and the date. We will also want cost here but for that leave placeholder i.e \$X and I will add it once the project is done.
- We need to also note that we had the `EpiSewer` source code embedded as a resource.

#### Documentation driven development

- We want to replicate the modelling in the readme of the package in our readme
- To begin this process we want to have text without code i.e Introduction is our example for the readme (not the checksums)
- To get ready for modelling we want to get the example data EpiSewer uses into our Julia package in a format we can load it in from (`ComposableTuringModels.jl` has examples of this).
- We will also want to add packages i.e `DataFramesMeta.jl` and `AlgebraOfGraphics.jl` that we will need for manipulating data and plotting it (as well as `PairsPlot.jl`).
- We then want to make a page in the documentation of the package (perhaps) the overview tutorial section or similar) that has a table of all the model components that EpiSewer has (I imagine giving them their R names but these will all be things with stan functions)
- We then want to explore `ComposableTuringIDModels.jl` and find the components it has that match elements of the `EpiSewer` model
- We now want to make github issues for the components that don't have counter parts in the `EpiAware.jl` ecosystem. These should be small modules that fit the `ComposableTuringIDModels.jl` patterns.
- Note that we don't need the custom discretisation tools Adrian uses here as we have those in `CensoredDistributions.jl` to make use of.

#### Component development

- For each of the small components we should implement it as a `ComposableTuringIDModels.jl` compatible `Struct`. We to have unit tests for these which test against functionality. We will also want to add them to our AD scenario testing suite. As implementing it is imperative to run the tests and confirm i.e AD support works.
- We should also have some small integration unit tests that test our new components in simple `ComposableTuringIDModels.jl`
- Make sure to sense check the new components against the the `EpiSewer` target components where possible. Usually it is best to do this by writing small scratch stan models (potentially saving the as `.gitignore` files in `.resouces` for safe keeping.). You can use generated quantities to test the eval. Generally we shoulnd't need to test gradients against each other here.

#### Implementing the examples

- We should now be able to implement the `EpiSewer` README modelling example. We want to use the style of composable model building used in the `ComposableTuringIDModels.jl` where we compose components with that ecosystem with the new ones we have made in this package cleanly.
- To confirm this model works we should then fit it to `EpiSewer` data - upgrading the NUTs settings as needed to ensure a clean and convergent fit (use 2 chains and 2 cores).
- We can then make each of the plots in the `EpiSewer` readme using our new model.
- We also want add a `PairsPlot.jl` that shows both the posterior and the prior densities for key parameters (likely not the latent model over time parameters for brevity).

#### Development process

- We want to make heavy use of subagents with the main agent just being a dispatcher into subagents.
- Generally we want to dispatch and agent to do a task (often one of these bullets), then another agent to review it against that bullet and earlier bullets.
- You should repeat this process until the review agent is happy. It is key that package tests are passing at all times and precommit is clean locally.
- Each agent should commit their task as a single commit (for those responding to the review including the correction points). It is fine to commit to main for this project.
- After each subtitle section is complete dispatch a review agent to double check the work from that section is complete and that CI is passing on the GitHub remote (check locally tests first to give CI time to run). You may start implementation/review loops as needed to deal with issues.
- After each subtitle section, also run `EpiAwarePackageTools.update(pkgdir(EpiSewer))` to re-apply the managed standard and report drift against the current kit, before dispatching the review agent.
- Review agents should make sure to check against `EpiAwarePackageTools.jl` enforced as well as written (i.e. in the documentation).
- You should make sure to make a todo list to do your manage the next step, then check this document again. Updating/clearing the todo list each time.
- As a guide we are aiming to highlight the `ComposableTuringIDModels.jl` ecosystem so we are aiming for full coverage of the model implemented here but with as few new components as possible i.e. we wish to reuse as much of our current ecosystem as we can.
- Note the style/names/interface is all drawn from `ComposableTuringIDModels.jl` and should be modular so i.e no `EpiSewer` wrapper function no none `Struct` interface elements or wrappers over multipe modules.
- It is important this work is async - don't try and access folders out of this one or take destructive actions recursively.
- If subagents stalled or fail feel free to restart those workers as needed.
