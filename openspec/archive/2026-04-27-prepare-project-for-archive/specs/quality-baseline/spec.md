## ADDED Requirements

### Requirement: Canonical quality commands are defined and green on mainline
The repository SHALL define a canonical verification baseline for formatting, linting, typing, tests, and package buildability. The mainline repository state MUST pass that baseline.

#### Scenario: Maintainer verifies repository health
- **GIVEN** a maintainer runs the documented baseline verification commands
- **WHEN** they execute the repository quality workflow
- **THEN** the formatting, linting, typing, CPU-safe test, and build checks SHALL succeed
- **AND** those commands SHALL match the project’s documented local and CI quality gates

### Requirement: Core quality workflows fail truthfully
GitHub workflows that represent required quality gates MUST fail on real errors and MUST NOT suppress core failures with unconditional success-shaped fallbacks.

#### Scenario: A quality regression reaches CI
- **GIVEN** a formatting, linting, typing, testing, or packaging regression is introduced
- **WHEN** the relevant CI workflow runs
- **THEN** the workflow SHALL report the failure clearly
- **AND** the regression SHALL not be masked by unconditional `|| true` patterns on core quality checks

### Requirement: Toolchain declarations are internally consistent
Repository configuration, editor guidance, CI workflows, and development-container settings SHALL not recommend conflicting formatters, linters, or version stories for the same workflow.

#### Scenario: Maintainer compares local and CI setup
- **GIVEN** a maintainer reviews repository tooling configuration
- **WHEN** they compare project metadata, editor settings, devcontainer settings, and CI workflow definitions
- **THEN** the repository SHALL describe one intentional formatting/lint/type-checking toolchain
- **AND** local guidance SHALL not contradict the CI-enforced path

### Requirement: Maintenance posture favors high-signal automation
The repository SHALL keep only automation that meaningfully protects its low-maintenance final state. Workflow breadth, matrix size, and non-blocking audits MUST be justified by clear maintenance value.

#### Scenario: Maintainer reviews automation scope
- **GIVEN** the repository is being prepared for a lower-maintenance phase
- **WHEN** a maintainer reviews active workflows and hooks
- **THEN** each retained automation path SHALL have a clear protective purpose
- **AND** ornamental or low-signal automation SHALL have been removed or downgraded from the critical path
