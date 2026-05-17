---
title: Academy
description: Whitepaper-style learning path for Triton Fused Ops
---

# Academy

The academy is the narrative route through the repository. Instead of listing APIs in isolation, it explains how the system is composed and why each piece exists.

## Academy map

| Sequence | Route | Why it comes here |
| :-- | :-- | :-- |
| 1 | [System overview](/en/academy/system-overview) | Understand the library as a layered system before focusing on one part |
| 2 | [Kernel Families](/en/kernel-families/) | Learn each user-facing operation in the language of workload, contracts, and evidence |
| 3 | [Architecture Lab](/en/architecture-lab/) | Inspect module seams, public exports, and runtime contracts |
| 4 | [Guides](/en/guides/) | Move from understanding to usage, measurement, and integration |
| 5 | [Reference & Research](/en/reference-research/) | Place the repo in the broader inference and kernel-systems conversation |

## Three ways to read it

### For evaluators

Start with the [system overview](/en/academy/system-overview), then jump to [runtime contracts](/en/architecture-lab/runtime-contracts). This path is optimized for reviewers asking whether the implementation is disciplined.

### For kernel engineers

Read [Kernel Families](/en/kernel-families/) next, then compare the descriptions against `triton_ops.kernels` and `triton_ops.reference`. This path is optimized for contributors who need to understand exactly where fusion, reference math, and validation join.

### For performance practitioners

Read the system overview, then move directly to the [performance guide](/en/guides/performance) and the [research notes](/en/reference-research/). This path is optimized for readers who care about Benchmarking, Auto-Tuning, and Performance metrics.

## What the academy emphasizes

- kernel families are the user-facing unit of reasoning,
- Benchmarking is evidence curation, not decoration,
- Auto-Tuning is a bounded subsystem with a latency-focused job,
- Performance metrics are derived only when the problem shape is explicit.
