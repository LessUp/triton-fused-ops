---
title: Architecture Lab
description: Module seams, public exports, and runtime contracts in Triton Fused Ops
---

# Architecture Lab

The architecture lab is the implementation-facing route through the codebase. It complements the academy by naming the actual modules, boundaries, and contracts that contributors need to respect.

## What to inspect here

- the [module map](/en/architecture-lab/module-map) for the repository’s structure,
- the [runtime contracts](/en/architecture-lab/runtime-contracts) for the validation and error model,
- the public exports that separate stable APIs from private Triton details.

## Why this section exists

A kernel library can look small on the surface while still hiding fragile coupling internally. This section is designed to show the opposite: the public exports are small, validation is centralized, and the surrounding support code explains rather than obscures the runtime path.

## Reading note

Start with the module map if you are orienting yourself. Jump directly to runtime contracts if you are patching an existing Kernel family or adding a new one and need to know what the launchers are allowed to assume.
