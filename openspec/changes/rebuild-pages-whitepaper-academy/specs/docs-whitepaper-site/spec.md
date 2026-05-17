## ADDED Requirements

### Requirement: The site SHALL present a bilingual whitepaper landing experience
The GitHub Pages site SHALL provide locale-specific landing pages for English and Simplified Chinese that introduce the repository as a technical whitepaper and architecture showcase, summarize the project’s core value, and direct readers into deeper documentation paths.

#### Scenario: Reader opens the English landing page
- **GIVEN** a reader navigates to `/en/`
- **WHEN** the page loads
- **THEN** the page MUST present a whitepaper-style hero, key project evidence, and links to deeper architecture and academy content

#### Scenario: Reader opens the Chinese landing page
- **GIVEN** a reader navigates to `/zh/`
- **WHEN** the page loads
- **THEN** the page MUST present the same whitepaper structure in Simplified Chinese with localized navigation and calls to action

### Requirement: The site SHALL guide readers through an academy-style information architecture
The site SHALL organize primary navigation and sidebars around staged learning sections that move from overview to deeper architecture, engineering guidance, and references.

#### Scenario: Reader explores top-level navigation
- **GIVEN** a reader is on any docs page
- **WHEN** they inspect the primary navigation
- **THEN** they MUST see section hubs that communicate a progressive learning flow rather than a flat manual layout

#### Scenario: Reader opens a section hub
- **GIVEN** a reader navigates to a section landing page
- **WHEN** the hub renders
- **THEN** it MUST explain who the section is for, what questions it answers, and which pages should be read next

### Requirement: The site SHALL preserve a predictable GitHub Pages documentation baseline
The site SHALL continue to build from the repository `docs/` directory through VitePress with locale-aware routing, local search, clean URLs, and GitHub Pages base-path handling.

#### Scenario: Repository docs are built for GitHub Pages
- **GIVEN** the Pages workflow configures a base path
- **WHEN** the docs build runs
- **THEN** the generated site MUST respect the configured base path and locale routing without broken navigation links

#### Scenario: Reader uses local search
- **GIVEN** a reader opens the docs search UI
- **WHEN** they query for a kernel family, architecture module, or benchmarking concept
- **THEN** the site MUST return results from the reorganized pages
