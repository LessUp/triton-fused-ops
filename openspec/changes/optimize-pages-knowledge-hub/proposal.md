## Why

The GitHub Pages site currently mixes technical knowledge with repository-facing noise, and several documentation pages drift away from the repository's actual exported APIs and runtime constraints. The published site should act as a bilingual knowledge hub rather than a mirror of root markdown files or a place for changelog-style content.

## What Changes

- Reframe GitHub Pages into a bilingual technical knowledge hub with cleaner landing pages, grouped navigation, and stronger visual hierarchy.
- Remove changelog, contributing, and other repository-process pages from the published Pages surface.
- Rewrite key documentation pages so they match the current repository code, especially around benchmark APIs, quantization helpers, validation rules, and integration guidance.
- Add missing knowledge coverage for models, validation contracts, and error semantics.

## Impact

- Affected areas: `_config.yml`, site assets, published markdown under `docs/`, sitemap/robots behavior, and OpenSpec artifacts for project presentation.
- User-facing impact: GitHub Pages becomes a cleaner bilingual technical site focused on maintained knowledge only.
