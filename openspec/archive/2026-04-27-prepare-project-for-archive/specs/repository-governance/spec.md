## ADDED Requirements

### Requirement: OpenSpec is the sole active change-management system
The repository SHALL use OpenSpec as its only active specification and change-management system in the mainline tree. Parallel planning or requirements systems MUST be removed from the mainline repository once any remaining useful intent has been migrated or intentionally discarded.

#### Scenario: Repository process audit after cleanup
- **GIVEN** a maintainer inspects the repository’s process and planning directories
- **WHEN** they review the active change-management assets
- **THEN** `openspec/` SHALL be the only active specification system in the mainline tree
- **AND** deprecated parallel systems such as `.kiro/` SHALL not remain in the mainline repository

### Requirement: Governance files are purposeful and project-specific
The repository SHALL keep only a minimal set of governance and instruction documents that directly support this project’s maintenance, AI collaboration, and contributor flow. Governance files MUST contain project-specific guidance and MUST NOT be retained as generic boilerplate.

#### Scenario: Governance document review
- **GIVEN** a maintainer opens the repository root and `.github/`
- **WHEN** they review governance and instruction files
- **THEN** the repository SHALL expose a purposeful set of project-specific documents
- **AND** redundant, low-signal, or generic governance files SHALL have been removed from the mainline tree

### Requirement: Change history is curated for end users
The repository SHALL expose a curated change history that reflects meaningful release or user-visible evolution. Low-value changelog fragments, iteration notes, or shallow historical noise MUST be removed from the mainline repository.

#### Scenario: User reviews project history
- **GIVEN** a user looks for release history in the repository
- **WHEN** they open the maintained changelog surface
- **THEN** they SHALL find meaningful version or release information
- **AND** they SHALL not have to navigate redundant fragment files to understand the project history
