## ADDED Requirements

### Requirement: Non-trivial work starts from an active OpenSpec change
Non-trivial repository work SHALL begin with an active OpenSpec change that includes proposal, design, and tasks before implementation or autonomous execution begins.

#### Scenario: Preparing a substantial repository change
- **GIVEN** a maintainer plans a cross-file, behavioral, or workflow-affecting change
- **WHEN** they prepare to implement it or hand it to autopilot execution
- **THEN** an active OpenSpec change SHALL already exist
- **AND** the change SHALL contain the artifacts required for implementation

### Requirement: AI execution flow is explicitly documented
The repository SHALL define a project-specific AI execution flow covering OpenSpec proposal/apply, `/review`, subagent usage, long-running execution, and the conditions for enabling autopilot or yolo-style execution.

#### Scenario: Maintainer chooses an execution mode
- **GIVEN** a maintainer is deciding how to run work with AI assistance
- **WHEN** they consult the project workflow guidance
- **THEN** they SHALL find clear rules for when to use planning, implementation, review, subagents, and autonomous execution
- **AND** the guidance SHALL discourage high-cost modes when lower-cost modes are sufficient

### Requirement: Repository-local automation is deterministic
The repository SHALL define deterministic local automation for touched files, including formatting and linting behavior that can run independently of any one assistant.

#### Scenario: Developer edits Python files locally
- **GIVEN** a developer changes one or more Python files
- **WHEN** they rely on the documented local automation and validation workflow
- **THEN** the repository SHALL provide a deterministic formatting and lint path for those files
- **AND** the same commands SHALL remain runnable outside a single AI tool integration

### Requirement: Language tooling and context integrations are intentionally minimal
The repository SHALL document a recommended language-tooling stack and SHALL keep heavyweight context integrations opt-in unless they provide a clear repository-specific benefit.

#### Scenario: Maintainer sets up local coding environment
- **GIVEN** a maintainer configures their editor and assistant tooling for this repository
- **WHEN** they follow the project’s setup guidance
- **THEN** they SHALL find a recommended LSP/editor toolchain that matches the repository’s Python workflow
- **AND** they SHALL find that optional MCP-style integrations are justified explicitly rather than enabled by default
