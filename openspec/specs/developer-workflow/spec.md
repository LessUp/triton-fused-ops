# Developer Workflow - Specification

## Overview

This spec defines the repository-level development flow used to keep implementation, documentation, and automation aligned.

## Requirements

### Requirement: Non-trivial work requires an active OpenSpec change
Non-trivial repository work SHALL begin with an active OpenSpec change that contains proposal, design, specs (if required), and tasks before implementation starts.

#### Scenario: Starting a cross-file change
- **WHEN** a maintainer plans a change that affects behavior, architecture, workflows, or multiple files
- **THEN** they SHALL create or select an active OpenSpec change first
- **AND** they SHALL complete required artifacts before implementation/autopilot execution

### Requirement: Task-driven implementation
Implementation SHALL follow `tasks.md` order and SHALL update task checkboxes as work is completed.

#### Scenario: Working through a change
- **WHEN** a maintainer implements a pending OpenSpec change
- **THEN** they SHALL execute work through task items in `tasks.md`
- **AND** they SHALL mark each task complete immediately after finishing it

### Requirement: Review before merge
Changes SHALL include an explicit review step and required validation checks before merge.

#### Scenario: Preparing a pull request for merge
- **WHEN** implementation is complete
- **THEN** the maintainer SHALL run repository validation commands and perform review
- **AND** the change SHALL not be merged without passing required quality checks
