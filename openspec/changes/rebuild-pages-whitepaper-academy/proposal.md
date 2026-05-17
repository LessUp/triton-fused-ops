## Why

The current GitHub Pages site already uses VitePress, but it still reads like a conventional project manual rather than a high-conviction technical whitepaper and architecture showcase. Its information architecture is shallow, the homepage is visually uneven, and theme-sensitive figures such as inline SVG benchmarks do not remain legible across light and dark appearance modes.

This change rebuilds the documentation site around a stricter academy-style reading path for evaluators and advanced developers, while aligning the underlying docs engineering baseline with the proven `kimi-cli` Pages stack and simplifying long-term maintenance.

## What Changes

- Rebuild the docs site around a whitepaper + academy narrative, with a stronger landing page, guided reading paths, and bilingual section hubs.
- Normalize the docs engineering baseline to the same VitePress 1.5 + local search + Mermaid + LLMs plugin + GitHub Pages deployment structure used by `/home/shane/dev/kimi-cli`.
- Replace the current ad-hoc visual treatment with a reusable documentation design system covering layout tokens, cards, figures, callouts, architecture panels, and benchmark visualizations.
- Fix dual-theme rendering bugs by making all charts, diagrams, and SVG-like figures explicitly aware of light/dark appearance tokens.
- Rewrite and expand core content so the site better explains kernel families, validation boundaries, benchmarking methodology, architecture seams, and related research/projects.
- **BREAKING**: Replace the existing docs information architecture and navigation tree without preserving legacy URL compatibility.

## Capabilities

### New Capabilities
- `docs-whitepaper-site`: A bilingual GitHub Pages experience that presents the repository as a guided technical whitepaper and architecture academy.
- `docs-dual-theme-visuals`: A visual system that guarantees readable, professional figures, diagrams, and charts in both light and dark appearance modes.
- `docs-academy-content`: A deeper content model that adds technical primers, architecture walkthroughs, benchmark methodology, related work, and evolution notes for advanced readers.

### Modified Capabilities
- None.

## Impact

- Affected code: `docs/.vitepress/**`, `docs/package.json`, `docs/index.md`, `docs/en/**`, `docs/zh/**`, `docs/scripts/**`
- Affected automation: `.github/workflows/pages.yml`
- Affected public surface: GitHub Pages navigation, landing pages, document hierarchy, internal links, and documentation visuals
- Dependencies/systems: VitePress theme layer, Mermaid rendering, Chart.js visualizations, GitHub Pages deployment pipeline
