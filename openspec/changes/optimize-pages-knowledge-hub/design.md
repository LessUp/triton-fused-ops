## Context

The repository already has a bilingual docs tree, but the Pages experience is not yet organized as a real knowledge system:

- root markdown files are still publishable on Pages even though they are repository artifacts rather than documentation pages,
- the site navigation is shallow and does not expose a strong information architecture,
- custom CSS and JS are present in the repository but are not connected to the Just the Docs theme,
- some API and guide pages describe symbols or workflows that are not actually exported or supported by the current code.

## Goals

- Publish only technical knowledge pages on GitHub Pages.
- Keep English and Chinese documentation in parallel.
- Improve the visual hierarchy without replacing the existing theme.
- Align key documentation pages with the current codebase.

## Decisions

### 1. Keep Just the Docs, but wire in real custom styling

The repository will keep the existing Just the Docs theme and attach custom CSS/JS through `_includes/head_custom.html` and `_includes/footer_custom.html`. This keeps the site easy to maintain while allowing a more intentional visual layer.

### 2. Use a grouped bilingual knowledge architecture

The site will expose four knowledge groups per language:

- Getting Started
- API Reference
- Guides
- Internals

Each group gets its own landing page so the sidebar reflects a clear learning path.

### 3. Exclude repository-process files from the published site

`README`, `CHANGELOG`, `CONTRIBUTING`, `AGENTS`, `CLAUDE`, and the local `changelog/` directory will be excluded from Pages generation. The published site should remain focused on technical knowledge only.

### 4. Prefer code-accurate docs over speculative integration claims

Integration, benchmark, and quantization pages will describe the currently shipped classes and functions. When framework integrations require adapter work outside the repository, the docs will say so directly instead of presenting speculative drop-in support.

## Risks

- Rewriting docs into a more curated form could remove some discoverable context. This is acceptable because the goal of Pages is focused knowledge delivery, not repository archaeology.
- Stronger visual styling could fight the theme if implemented too broadly. This is mitigated by limiting the work to additive CSS and a small amount of DOM enhancement.
