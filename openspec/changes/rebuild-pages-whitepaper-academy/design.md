## Context

`triton-fused-ops` already ships a VitePress-based GitHub Pages site, and the docs deployment workflow is close to the `kimi-cli` baseline: VitePress 1.5, local search, Mermaid, the LLMs plugin, locale-aware routing, and a Pages workflow that injects `VITEPRESS_BASE`. The gap is not framework choice; it is product design and information architecture.

The current site mixes a technical-whitepaper intent with a component-heavy homepage, a conventional API/Guides/Internals split, and uneven visual semantics. The result is readable but not memorable. Several visuals also rely on hard-coded colors that degrade in light mode, especially inline benchmark SVGs. The user explicitly wants an aggressive reset: no backward compatibility, stronger “academy / architecture showcase” positioning, and a presentation quality that holds up under interview scrutiny.

The reference repo `/home/shane/dev/kimi-cli` establishes the preferred engineering baseline: a lean VitePress setup, a restrained theme entrypoint, CSS-token-driven customization, bilingual nav/sidebar trees, and a Pages workflow designed for GitHub Pages base-path correctness. This redesign should preserve that baseline while dramatically increasing the editorial depth and visual sophistication required by this repository’s GPU-kernel domain.

## Goals / Non-Goals

**Goals:**

- Keep the GitHub Pages stack aligned with the `kimi-cli` baseline where it matters: package scripts, VitePress configuration patterns, locale routing, Pages deployment, and theme organization.
- Rebuild the site IA into a whitepaper / academy flow that works for two main audiences: evaluators who want a sharp project overview, and advanced developers who want implementation depth.
- Introduce a durable visual system with reusable, theme-safe components for hero sections, stat strips, kernel cards, architecture maps, process diagrams, and benchmark figures.
- Expand the docs with stronger technical narrative: system overview, module seams, kernel-family deep dives, benchmarking methodology, related work, and forward-looking evolution notes.
- Maintain bilingual parity between English and Simplified Chinese for all first-class landing and academy pages.

**Non-Goals:**

- No attempt to preserve legacy documentation URLs or sidebars.
- No change to the Python package API or kernel implementation behavior.
- No new runtime docs backend, search service, or analytics platform.
- No speculative marketing microsite outside the existing `docs/` GitHub Pages site.

## Decisions

### 1. Keep the `kimi-cli` docs engineering baseline and remove unnecessary divergence

**Decision:** Preserve VitePress 1.5, local search, Mermaid, the LLMs plugin, and the GitHub Pages deployment shape already shared with `kimi-cli`. Simplify the theme layer so the baseline stays recognizable: a single `config.ts`, a minimal `theme/index.ts`, and a token-heavy `style.css` that drives most customization.

**Rationale:** The user asked for the same advanced framework and implementation design as the local `kimi-cli` Pages site. Since the current repo already converged on that stack, the highest-value move is to eliminate accidental complexity and align naming, layout patterns, and maintenance posture rather than swap frameworks.

**Alternatives considered:**

- **Move to another docs framework (Astro/Next/Nextra):** Rejected because it breaks the “same stack as `kimi-cli`” requirement and adds migration cost without solving the core IA/content problem.
- **Keep the current custom-component-heavy approach untouched:** Rejected because it makes the site harder to evolve and still does not solve the editorial and theme-quality issues.

### 2. Replace the current hierarchy with an academy-style reading model

**Decision:** Reorganize both locales around a staged reading path:

1. `Overview` / `导读`
2. `Academy` / `学院`
3. `Kernel Families` / `算子族`
4. `Architecture Lab` / `架构实验室`
5. `Engineering Guides` / `工程指南`
6. `Reference & Research` / `参考与研究`
7. `Release Notes` / `发布说明`

Each section gets a landing page that explains what readers will learn, who it is for, and where to go next.

**Rationale:** The current `Getting Started / API / Guides / Internals / References` taxonomy is serviceable but flat. An academy model better matches the user’s “technical whitepaper / project academy” goal and creates a clearer reading path from shallow introduction to deep architecture and research context.

**Alternatives considered:**

- **Minor nav cleanup only:** Rejected because it would not produce the “dimension reduction strike” effect the user wants.
- **Pure whitepaper single-page site:** Rejected because the project still needs navigable reference material, not just a narrative landing page.

### 3. Introduce a tokenized design system and figure grammar

**Decision:** Build a docs-specific design system using CSS custom properties and small reusable Vue components for:

- whitepaper hero blocks
- editorial section headers
- metric/stat grids
- reader-journey cards
- architecture maps
- evidence callouts
- bibliography / related-work panels
- figure frames with captions and status badges

All figure primitives use semantic tokens (`surface`, `line`, `accent`, `success`, `warning`, `muted`) derived from VitePress theme variables instead of hard-coded light/dark colors.

**Rationale:** The current theme mixes strong ideas with one-off visual treatments. A tokenized system allows a more premium look while making dual-theme support deterministic and maintainable.

**Alternatives considered:**

- **Only restyle Markdown prose:** Rejected because the user explicitly wants richer diagrams, commercial-grade visuals, and a more distinctive landing experience.
- **Render everything as images:** Rejected because static assets are harder to maintain, harder to localize, and the current bug report specifically targets appearance-mode failures.

### 4. Convert all architecture and benchmark visuals to theme-aware render paths

**Decision:** Replace hard-coded inline SVG colors and brittle custom figure markup with one of three render paths:

- tokenized inline SVG inside reusable figure wrappers
- Mermaid diagrams styled through shared theme variables
- Chart.js components that derive palette, grid, labels, and tooltip colors from `useData().isDark`

Every visual gets a caption, legend, and fallback textual explanation in the surrounding prose.

**Rationale:** This directly addresses the light/dark rendering bug while improving accessibility and consistency. It also keeps visuals native to the docs stack instead of introducing a separate illustration pipeline.

**Alternatives considered:**

- **Leave existing inline SVGs and tweak colors manually:** Rejected because it does not scale and would reintroduce regressions on future pages.
- **Use only Mermaid for everything:** Rejected because benchmark charts and editorial illustrations need more control than Mermaid provides.

### 5. Treat content as first-class architecture, not filler around API pages

**Decision:** Rewrite the landing pages and section hubs around explicit reader questions:

- What problem does this project solve?
- Which kernel family should I care about?
- How is correctness established?
- Where do validation, compute references, autotuning, benchmarking, and performance metrics fit?
- What prior art and research context support these design choices?

Add dedicated sections for benchmark methodology, comparative analysis, related open-source projects, citations, and “evolution thinking” to frame future architectural seams.

**Rationale:** The repository already has strong technical substance in README, CONTEXT, and code structure. The docs should surface that substance with more rigor so the site reads like an engineer-authored whitepaper instead of a thin landing page plus API stubs.

## Risks / Trade-offs

- **[Large doc churn]** → Mitigation: rebuild navigation and landing pages coherently in one change rather than mixing old and new IA.
- **[Bilingual parity drift]** → Mitigation: establish mirrored EN/ZH page trees and translate all new landing/academy pages before finishing.
- **[Over-designed UI hurts readability]** → Mitigation: keep the `kimi-cli` baseline restraint for global chrome and reserve visual richness for home/section/figure components.
- **[Theme regressions on new diagrams]** → Mitigation: centralize figure tokens and update all visual components to derive colors from the active appearance state.
- **[Search/index quality degrades during reorganization]** → Mitigation: ensure every hub page has strong headings, summaries, and stable descriptive titles for local search relevance.

## Migration Plan

1. Normalize docs scaffolding and navigation around the new section map.
2. Build the new theme tokens and shared components.
3. Rewrite English pages first to establish canonical structure and copy.
4. Mirror the new structure in Simplified Chinese with localized copy.
5. Remove superseded pages/components once replacement pages are linked.
6. Rebuild docs and validate theme rendering, internal links, and Pages output.

Rollback is simple because the site remains within the existing `docs/` VitePress deployment. Reverting the change restores the previous Pages site.

## Open Questions

- None blocking. The user explicitly asked for an aggressive no-compatibility rebuild, so the design assumes full freedom to replace the existing IA and visuals.
