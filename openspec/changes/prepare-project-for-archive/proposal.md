## Why

The repository has drifted away from a disciplined OpenSpec-driven workflow: overlapping specification systems, noisy and outdated documentation, over-designed project infrastructure, and inconsistent automation now obscure the actual value of the Triton kernels. Because the project is entering a completion-and-archive posture rather than an open-ended growth phase, it needs a focused convergence pass that removes low-value assets, restores architectural clarity, and leaves behind a stable, credible, low-maintenance final state.

## What Changes

- Consolidate the project onto a single OpenSpec-first governance model and remove parallel or low-value process systems from the mainline repository.
- Redesign repository-level documentation so that README, Pages, changelog, engineering docs, and contributor guidance are purposeful, current, and tightly scoped to this project.
- Introduce project-specific agent and AI workflow documentation (`AGENTS.md`, project-level `CLAUDE.md`, Copilot instructions, workflow guidance) that formalizes how OpenSpec, review, subagents, and long-running execution should be used on this repository.
- Simplify and harden engineering automation: hooks, CI/workflows, development environment configuration, and version/tooling alignment must favor reliable signal over noisy or redundant jobs.
- Audit the Python package, tests, and public claims; fix concrete defects and normalize formatting/lint issues so that the repository matches its published quality bar.
- Reposition GitHub Pages and repository metadata to act as a concise project landing surface rather than a shallow mirror of the README.

## Capabilities

### New Capabilities
- `repository-governance`: Defines the repository’s canonical process assets, governance files, cleanup policy, and archive-ready maintenance posture.
- `developer-workflow`: Defines the OpenSpec-first development flow, AI assistant usage model, review checkpoints, hooks, and local/cloud execution conventions for this project.
- `project-presentation`: Defines how README, GitHub Pages, changelog surfaces, and GitHub About metadata present the project to users.
- `quality-baseline`: Defines the repository’s minimal but strict quality gates across formatting, linting, typing, tests, and workflow signal quality.

### Modified Capabilities
- `core`: Tighten published expectations so that documented capabilities, compatibility claims, and user-facing guidance reflect the repository’s actually supported and verified behavior.

## Impact

- Affected areas: `openspec/`, `.github/`, `.claude/`, docs and site files, root project documentation, change/changelog assets, development environment config, and selected Python source/test files.
- Expected removals: redundant changelog entries, outdated or low-signal documents, parallel spec/process assets, and workflow/configuration noise that does not serve the project’s final maintenance posture.
- Expected GitHub-facing impact: repository About/description/topics, Pages positioning, and contributor/developer entry points become more intentional and easier to trust.
