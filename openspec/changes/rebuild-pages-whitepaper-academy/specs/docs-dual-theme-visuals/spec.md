## ADDED Requirements

### Requirement: Technical visuals SHALL remain legible in both appearance modes
All first-class diagrams, benchmark visuals, iconography, and figure-like illustrations SHALL use theme-aware colors and contrast levels that remain readable in light and dark modes.

#### Scenario: Reader switches from dark mode to light mode
- **GIVEN** a page contains diagrams or benchmark figures
- **WHEN** the site appearance changes from dark to light
- **THEN** text, lines, fills, legends, and emphasis states MUST remain legible without disappearing or blending into the background

#### Scenario: Reader opens a page with inline SVG content
- **GIVEN** a page renders inline SVG-based visuals
- **WHEN** the page is viewed in either appearance mode
- **THEN** the SVG strokes, fills, and labels MUST derive from theme-aware visual tokens instead of hard-coded mode-specific colors

### Requirement: Documentation figures SHALL follow a reusable editorial figure grammar
The site SHALL provide reusable visual components or styles for figure framing, captions, legends, callouts, and metric summaries so diagrams and charts read as a coherent system.

#### Scenario: Author adds a new architecture figure
- **GIVEN** a maintainer creates a new figure for the docs
- **WHEN** the figure is rendered
- **THEN** it MUST support a consistent frame, caption treatment, and semantic color vocabulary shared across the site

#### Scenario: Reader scans multiple technical pages
- **GIVEN** a reader moves between architecture, performance, and reference pages
- **WHEN** they encounter diagrams and highlight blocks
- **THEN** those visuals MUST feel like part of one consistent design system

### Requirement: Interactive benchmark charts SHALL adapt to the active theme
Interactive charts rendered through the docs theme SHALL derive axis colors, grid lines, labels, and tooltips from the active appearance state.

#### Scenario: Reader views a benchmark chart in dark mode
- **GIVEN** a benchmark chart is rendered in dark mode
- **WHEN** the chart initializes
- **THEN** its text, grid, tooltip, and series colors MUST match the dark theme contrast requirements

#### Scenario: Reader toggles appearance after chart initialization
- **GIVEN** a benchmark chart has already rendered
- **WHEN** the site appearance changes
- **THEN** the chart MUST refresh to the new theme palette without stale colors remaining on screen
