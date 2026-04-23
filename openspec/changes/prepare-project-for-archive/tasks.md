## 1. Governance convergence

- [x] 1.1 Remove parallel process systems and low-value mainline planning/history assets, including `.kiro/` and redundant changelog/doc fragments, while preserving only intentional OpenSpec and user-facing history surfaces. _(Requirements: RG1, RG3, PP4)_
- [x] 1.2 Rewrite the repository’s canonical governance/instruction files so `AGENTS.md`, root `CLAUDE.md`, and contributor workflow guidance are project-specific and aligned to the archive-ready maintenance posture. _(Requirements: RG2, DW2)_
- [x] 1.3 Align `openspec/specs/` and related repository-level process docs so future non-trivial work starts from an active OpenSpec change before implementation or autopilot execution. _(Requirements: RG1, DW1)_

## 2. AI workflow and tooling guidance

- [x] 2.1 Add project-level Copilot instructions and harmonize `.claude/` guidance so Claude, Copilot, Codex, `/review`, subagents, `/remote`, and `/research` are documented as one coherent operating model. _(Requirements: DW2)_
- [x] 2.2 Define and implement repository-local hooks plus editor guidance for deterministic touched-file formatting/linting that does not depend on one assistant integration. _(Requirements: DW3, QB3)_
- [x] 2.3 Standardize the recommended LSP/editor/tooling stack and document why MCP remains opt-in and minimal for this repository. _(Requirements: DW4, QB3)_

## 3. Presentation and documentation redesign

- [x] 3.1 Rewrite README and key documentation entry points so capability claims, prerequisites, and user journeys are credible, current, and mutually consistent. _(Requirements: PP1, PP4, CORE1, CORE2)_
- [x] 3.2 Redesign GitHub Pages home/navigation/configuration into a curated landing surface instead of a shallow README mirror. _(Requirements: PP1, PP2)_
- [x] 3.3 Update GitHub repository About metadata with a concise description, homepage URL, and relevant topics using `gh` so repository discovery aligns with the new positioning. _(Requirements: PP1, PP3)_

## 4. Automation and engineering simplification

- [x] 4.1 Simplify GitHub Actions so core quality checks fail truthfully, remove low-signal workflow behavior, and keep only maintenance-appropriate automation in the critical path. _(Requirements: QB2, QB4)_
- [x] 4.2 Align `pyproject.toml`, devcontainer/editor settings, hook behavior, and workflow tool choices around one consistent formatting/lint/type-check/build story. _(Requirements: QB1, QB3)_

## 5. Repository defect cleanup and final baseline

- [x] 5.1 Fix the current source/test formatting and lint drift, then resolve any additional repository defects discovered while making the new quality baseline truthful. _(Requirements: QB1, QB2)_
- [x] 5.2 Reconcile user-facing performance/compatibility claims with maintained evidence, tests, benchmarks, and explicit caveats. _(Requirements: CORE1, CORE2, PP1)_
- [x] 5.3 Run the canonical verification baseline, confirm the repository matches the final spec set, and leave the change ready for implementation closure and archive-oriented maintenance. _(Requirements: QB1, QB4, DW1)_
