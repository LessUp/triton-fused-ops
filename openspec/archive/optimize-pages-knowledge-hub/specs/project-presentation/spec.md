## ADDED Requirements

### Requirement: GitHub Pages publishes technical knowledge only

The GitHub Pages site SHALL publish curated technical documentation only, and SHALL exclude repository-process content such as changelogs, contributing instructions, and assistant workflow files.

#### Scenario: User browses the published site
- **WHEN** a user navigates the Pages site
- **THEN** they SHALL see technical knowledge pages only
- **AND** changelog-style or repository-process pages SHALL not appear in site navigation or generated page routes

### Requirement: GitHub Pages presents a bilingual knowledge architecture

The site SHALL provide parallel English and Chinese documentation organized into stable knowledge groups for onboarding, API reference, engineering guidance, and internals.

#### Scenario: User selects a language
- **WHEN** a user opens the English or Chinese docs root
- **THEN** they SHALL see the same information architecture in both languages
- **AND** the navigation SHALL expose grouped knowledge sections rather than a flat page list

### Requirement: Published API knowledge matches the maintained code

The user-facing API documentation SHALL reflect the functions, classes, validation rules, and exceptions that are actually present in the repository.

#### Scenario: User reads an API page
- **WHEN** a user consults documentation for kernels, quantization, autotuning, benchmark tooling, or validation behavior
- **THEN** the page SHALL describe currently implemented interfaces and constraints
- **AND** it SHALL not present speculative or unsupported root-level exports as supported public API
