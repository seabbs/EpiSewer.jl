# LLM-Assisted Development Process

## Aim

This package is an experiment to see if non-frontier models can be used to effectively port R packages to Julia when using composable elements built by frontier models.

## Methodology

- **Code harness**: pi coding agent with subagents (pi-subagents)
- **Language model**: DeepSeek V4 Flash via pi-agent (openrouter/~deepseek/deepseek-v4-flash-latest)
- **Date**: August 2026
- **Cost**: $X (placeholder — to be filled in once the project is complete)

## Review pass

- **Date**: 2026-08-15. **Harness**: pi coding agent with pi-subagents (same as the implementation pass).
- **Review model**: openrouter/~deepseek/deepseek-v4-flash-latest — the same non-frontier model as the implementation pass, so the experiment covers self-review.
- **Cost**: $Y (placeholder — to be filled in once the project is complete).
- **Instructions**: review each section of the replication prompt against the repository state; get CI green (quality suite, JET, AD, Documenter); correct drift from the EpiAwarePackageTools README standard; make the README worked example runnable code for the Documenter pipeline rather than saved plots and scripts; align the example data with the original EpiSewer example (sparse Monday/Thursday measurements over the same window); enforce ecosystem reuse over custom code; keep code and comments concise and elegant; work through heavy weak-worker subagent delegation with the review model as overseer, committing and pushing each fix.

**Operator bumps**: the implementation agent stopped work twice and had to be bumped by the operator — (1) after component development, leaving empty placeholder modules, an outdated model-components table, and the examples section unstarted (operator sent a "keep-going" prompt); (2) at the final checkpoint, stopping with CI still red and asking whether to continue.

## Frontier-model review pass

The experiment's premise is that a non-frontier model can port a package when the composable pieces are supplied by frontier models.
Testing that premise needs a frontier model to check the result, so a third pass audited the two non-frontier passes above.

- **Date**: 2026-08-17.
- **Harness**: Claude Code 2.1.233 (CLI), main agent plus `Task` subagents.
- **Review model**: Claude Opus 5 (1M-token context) for the main agent and every subagent.
- **Scale**: one main agent plus 2 subagents, ~515 model calls, ~261k output tokens and ~81 million input-side tokens (almost all of it prompt-cache reads), measured from the session transcripts. For comparison, the whole audit fits in a single context window of the model used.
- **Cost**: $Z (placeholder — to be filled in once the project is complete).
- **Instructions**: work through every bullet of the replication prompt and record its true status against the repository rather than against the previous passes' own session notes; fix the failing Enzyme AD CI jobs; and judge how closely the package tracks `ComposableTuringIDModels.jl` in style and in reuse.

### What the review found

Most bullets were delivered.
The composable ecosystem carried the port: five new structs (`LOD`, `LogNormalError`, `DigitalPCRError`, `MeasurementOutliers`, `FlowNormalize`) cover a model whose R original is built from about twenty components, and the rest is composition of existing `ComposableTuringIDModels.jl` pieces.
The failures cluster in four places.

1. **Numerical correctness of the assembled model.**
   The observation chain was assembled in the wrong order and the load-per-case prior was two orders of magnitude off the data scale.
   Both still ran, and both still passed every unit test, because a mis-ordered composition of correct components is a valid model.
   Only a sense-check of fitted `R_t` against a plausible range caught it, and that check was prompted by the operator rather than volunteered.
2. **Fidelity where the target's own source had to be read.**
   `MeasurementOutliers` was written as a two-component contamination mixture.
   EpiSewer's `outliers_estimate` is not a mixture: it draws additive spikes from a generalised extreme value distribution and scales them by the load per case.
   The component and the documentation agree with each other and disagree with the package being ported.
3. **Tests that exercise nothing.**
   Every component has an AD gradient scenario, and no scenario calls the component.
   Each one re-implements the component's log-density inside the fixture file, so the AD matrix reports on the fixture rather than on the package.
   The frontier-built reference package differentiates real model log-joints instead.
4. **Documentation drifting behind the code.**
   The model-components table described designs that were replaced within a few commits of being written: a flow normalisation rescaling by a reference flow, a coefficient-of-variation noise model attributed to the wrong component, and count-family error models offered for a continuous likelihood.
   Those rows were corrected in this pass.

Infrastructure was the largest single time sink across both non-frontier passes, and it produced the longest-lived failure.
The Enzyme AD jobs stayed red for two days over a missing `function_annotation = Enzyme.Const` on the backend constructors — a one-line annotation that `ComposableTuringIDModels.jl` already carries, with the reason written in a comment beside it.

The full bullet-by-bullet audit is kept with the other review notes in the gitignored `.resources/` directory.

## Source material

The original EpiSewer R package source code was embedded as a resource at `.resources/EpiSewer/`.

## The replication prompt

The full replication prompt that guided this project is preserved verbatim below. It lives in this page because the original `.resources/prompt.md` is not committed (`.resources` is gitignored).

```text
# Replication plan for EpiSewer

## Aim

I want to replicate the model functionality but not interface or other tools (i.e. plotting, docker etc) of the EpiSewer package using tools from the EpiAware.org GitHub Ecosystem

## Setup

- We want to first make sure we can install and run `EpiSewer`
- We will first set the Package up using `EpiAwarePackageTools.jl`.
- Add `.resources` to the `.gitignore
- Our core tools and dependencies will be `ComposableTuringIDModels.jl`, `CensoredDistributions.jl`, `EpiAwareADTools.jl`. This is on top of the tools package tools will bring.
- We know there are issues declaring as not part of the EpiAware.org ecosystem so we may need to make a few tweaks so that the docs show up in our samabbott.co.uk/EpiSewer.jl.
- We need to opt in to AD benchmarking
- We will want to flesh out what this package is i.e its readme folling EpiAware standards
- We need to add a clear derived from EpiSewer attribution
- We will want to make a documentation page that details the LLM process used to create the package. It should have this document in it as i.e. the prompt. It should start with the aim (experimenting to see if non-frontier models can be used to effectively language port packages when using composable elements built by frontier models). You should record in this document the code harness we are using as well the as the LLM model used and the date. We will also want cost here but for that leave placeholder i.e $X and I will add it once the project is done.
- We need to also note that we had the `EpiSewer` source code embedded as a resource.

## Documentation driven development

- We want to replicate the modelling in the readme of the package in our readme
- To begin this process we want to have text without code i.e Introduction is our example for the readme (not the checksums)
- To get ready for modelling we want to get the example data EpiSewer uses into our Julia package in a format we can load it in from (`ComposableTuringModels.jl` has examples of this).
- We will also want to add packages i.e `DataFramesMeta.jl` and `AlgebraOfGraphics.jl` that we will need for manipulating data and plotting it (as well as `PairsPlot.jl`).
- We then want to make a page in the documentation of the package (perhaps) the overview tutorial section or similar) that has a table of all the model components that EpiSewer has (I imagine giving them their R names but these will all be things with stan functions)
- We then want to explore `ComposableTuringIDModels.jl` and find the components it has that match elements of the `EpiSewer` model
- We now want to make github issues for the components that don't have counter parts in the `EpiAware.jl` ecosystem. These should be small modules that fit the `ComposableTuringIDModels.jl` patterns.
- Note that we don't need the custom discretisation tools Adrian uses here as we have those in `CensoredDistributions.jl` to make use of.

## Component development

- For each of the small components we should implement it as a `ComposableTuringIDModels.jl` compatible `Struct`. We to have unit tests for these which test against functionality. We will also want to add them to our AD scenario testing suite. As implementing it is imperative to run the tests and confirm i.e AD support works.
- We should also have some small integration unit tests that test our new components in simple `ComposableTuringIDModels.jl`
- Make sure to sense check the new components against the the `EpiSewer` target components where possible. Usually it is best to do this by writing small scratch stan models (potentially saving the as `.gitignore` files in `.resouces` for safe keeping.). You can use generated quantities to test the eval. Generally we shoulnd't need to test gradients against each other here.

## Implementing the examples

- We should now be able to implement the `EpiSewer` README modelling example. We want to use the style of composable model building used in the `ComposableTuringIDModels.jl` where we compose components with that ecosystem with the new ones we have made in this package cleanly.
- To confirm this model works we should then fit it to `EpiSewer` data - upgrading the NUTs settings as needed to ensure a clean and convergent fit (use 2 chains and 2 cores).
- We can then make each of the plots in the `EpiSewer` readme using our new model.
- We also want add a `PairsPlot.jl` that shows both the posterior and the prior densities for key parameters (likely not the latent model over time parameters for brevity).

## Development process

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
```
