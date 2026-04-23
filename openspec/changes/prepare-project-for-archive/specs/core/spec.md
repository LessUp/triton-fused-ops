## ADDED Requirements

### Requirement: Published capability claims are evidence-backed
User-facing project claims about supported environments, accuracy behavior, and performance improvements SHALL be backed by maintained code, tests, benchmarks, or explicit scope qualifiers.

#### Scenario: User evaluates project claims
- **GIVEN** a user reads README, Pages, or documentation claims about speedup, compatibility, or numerical behavior
- **WHEN** they compare those claims with the repository’s maintained validation surfaces
- **THEN** the claims SHALL be supported by current evidence or explicit caveats
- **AND** unsupported or unverifiable claims SHALL not remain in maintained user-facing documents

### Requirement: Runtime prerequisites are clearly bounded
Installation and usage guidance SHALL clearly distinguish CPU-safe verification from GPU-required functionality and SHALL state the required environment assumptions for actual kernel execution.

#### Scenario: User prepares an environment
- **GIVEN** a user follows the repository’s installation or quickstart guidance
- **WHEN** they determine whether they can validate or run the project on their machine
- **THEN** the guidance SHALL distinguish CPU-only checks from GPU-dependent execution paths
- **AND** the user SHALL understand the hardware and software prerequisites before attempting runtime execution
