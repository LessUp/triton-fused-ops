## Context

The repository currently presents a drifted operating model:

- OpenSpec exists, but there is no active change driving the work.
- A parallel `.kiro/` process tree remains in the mainline repository.
- There is no root `AGENTS.md`, no project-level `CLAUDE.md`, and no Copilot instruction file.
- GitHub Pages is heavily engineered but not clearly differentiated from the README.
- CI contains useful checks, but some core jobs suppress failure (`|| true`) and some environment guidance is internally inconsistent.
- Baseline quality is structurally close to green: CPU-safe tests and mypy pass, while current failures are concentrated in auto-fixable format/lint drift.

This change is cross-cutting: it affects governance, developer workflow, documentation, site presentation, automation, and selected source files. It is also intentionally biased toward repository closure and low-maintenance stability rather than feature growth.

Kernel math, numerical algorithms, and HBM access patterns are **not** the primary target of this change. Existing fused-kernel behavior remains the product core; this effort is about making the surrounding repository honest, coherent, and maintainable.

## Goals / Non-Goals

**Goals:**
- Converge the repository onto one canonical specification and change-management system: OpenSpec.
- Remove low-value or redundant assets from the mainline tree and leave behind a smaller, higher-signal repository.
- Define a project-specific AI/developer workflow covering OpenSpec, `/review`, subagents, long-running execution, and archive-ready maintenance.
- Align README, Pages, GitHub About metadata, and user-facing docs around a single credible project narrative.
- Simplify automation so local hooks, CI, and development environment configuration give consistent and truthful feedback.
- Fix concrete repository defects discovered during the audit, including format/lint drift and workflow anti-patterns.

**Non-Goals:**
- Adding new Triton kernels or expanding the public API surface.
- Re-architecting numerical kernels or changing validated precision behavior.
- Introducing heavyweight always-on MCP infrastructure without a demonstrated gap.
- Turning this repository back into a fast-moving feature-development project.

## Decisions

### 1. OpenSpec becomes the sole active process layer

**Decision:** Keep `openspec/` as the only active requirements/change-management system in the mainline repository. Remove parallel systems such as `.kiro/` after extracting any remaining useful intent into OpenSpec artifacts or final docs.

**Why:** Multiple planning systems create contradictory truth sources and encourage process drift.

**Alternatives considered:**
- **Keep both OpenSpec and `.kiro/`:** rejected because it preserves ambiguity.
- **Archive `.kiro/` into a legacy folder:** rejected because the user explicitly prefers aggressive removal from mainline, and git history already preserves it.

### 2. Use a minimal instruction triad for AI collaboration

**Decision:** Standardize on:
- `AGENTS.md` for repo-wide task execution guidance,
- root `CLAUDE.md` for Claude-specific working norms,
- `.github/copilot-instructions.md` for Copilot/GitHub-native guidance.

These files must be project-specific and reference the same OpenSpec-first flow.

**Why:** Each tool benefits from a native entry point, but the repository should not accumulate endless overlapping guidance files.

**Alternatives considered:**
- **One universal doc only:** rejected because some tools discover different convention files.
- **Many tool-specific docs/configs:** rejected because that recreates the sprawl we are trying to remove.

### 3. Favor a low-context tool stack over heavyweight MCP

**Decision:** Do not add project-level MCP by default. Prefer built-in `gh`, OpenSpec CLI, review flows, subagents, and existing skills. MCP should be added only if a concrete missing capability remains after cleanup.

**Why:** This repository is heading toward completion, and context-heavy integrations increase operational overhead without clear benefit.

**Alternatives considered:**
- **Preconfigure several MCP servers:** rejected due to context cost, maintenance burden, and marginal value for a mostly-finish-line project.
- **No guidance at all:** rejected because the repo still needs a documented tool-selection policy.

### 4. Standardize editor and language tooling at the repository level

**Decision:** Treat LSP as an editor/runtime concern rather than an assistant-specific feature. Standardize on Python language intelligence plus Ruff and mypy through repo-local editor/devcontainer guidance.

Recommended stack:
- VS Code / compatible editors: Pylance (or Pyright-compatible engine)
- Ruff for formatting + linting
- mypy for repository type-check gates

**Why:** Copilot, Claude, and Codex do not all consume the same language-server channel directly, but they all benefit when the repository’s editor configuration, diagnostics, and commands are coherent.

**Alternatives considered:**
- **Assistant-specific LSP setups:** rejected because they do not generalize well across tools.
- **Keep Black + Ruff split formatting story:** rejected because the repository already leans toward Ruff and the current setup is internally inconsistent.

### 5. Pages and README must share one positioning, not duplicate content

**Decision:** Reframe GitHub Pages as a concise project landing site with proof, positioning, integration entry points, and selective deep links. README remains the repository entry point; Pages becomes the external showcase surface.

**Why:** A copy-moved README does not justify a separate site and does not help discovery or trust.

**Alternatives considered:**
- **Keep Pages as a README mirror:** rejected because it adds maintenance without user value.
- **Delete Pages entirely:** rejected because a focused landing page still helps discovery and GitHub About linking.

### 6. Quality gates must fail truthfully and remain closure-friendly

**Decision:** Keep a strict but narrow verification baseline:

```text
ruff format --check .
ruff check .
mypy triton_ops/
pytest tests/ -v -k "not cuda and not gpu" --ignore=tests/benchmarks/
python -m build
```

Core quality jobs must not use unconditional success fallbacks. Workflow scope should be reduced to checks that meaningfully protect the repository’s final maintained state.

**Why:** A smaller truthful pipeline is more valuable than a larger noisy one.

**Alternatives considered:**
- **Keep all current workflow embellishments:** rejected because they create noise and weak signal.
- **Remove most CI:** rejected because an archive-ready repo still needs credibility.

## Risks / Trade-offs

- **[Risk] Aggressive deletions remove discoverable context** → **Mitigation:** keep only high-value final docs in-tree; rely on git history and OpenSpec archive for discarded detail.
- **[Risk] Reducing workflows may hide niche regressions** → **Mitigation:** preserve one strong verification path and document optional GPU/manual checks separately.
- **[Risk] Consolidating AI guidance may miss tool-specific nuance** → **Mitigation:** use a minimal shared flow plus only the few native instruction files that tools actually consume.
- **[Risk] Tightening claims may make the project look less ambitious** → **Mitigation:** prefer credibility and evidence over inflated messaging.

## Migration Plan

1. Establish the canonical OpenSpec change and final governance model.
2. Remove redundant process systems and prune low-value docs/changelog artifacts from the mainline tree.
3. Rewrite the project’s human and AI instruction surfaces (`AGENTS.md`, `CLAUDE.md`, Copilot instructions, contributing/workflow guidance).
4. Rework README, Pages, and GitHub metadata so all external entry points align.
5. Simplify CI/hooks/editor configuration and align tooling versions/story.
6. Fix repository defects surfaced by the verification baseline, then run the final verification set.

Rollback is straightforward because the work is repository-structure and configuration focused; individual deletions or rewrites can be restored from git history if needed.

## Open Questions

- Whether bilingual documentation should remain fully mirrored or be reduced to a smaller curated Chinese surface if parity proves too costly during cleanup.
- Whether any current Pages-specific performance/audit steps still provide enough value to justify their runtime and maintenance cost.
