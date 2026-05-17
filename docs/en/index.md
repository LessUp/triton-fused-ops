---
layout: page
title: Triton Fused Ops
description: Technical whitepaper landing page for Triton Fused Ops
---

<script setup>
import WhitepaperHero from '@theme/components/WhitepaperHero.vue'
import ReaderTracks from '@theme/components/ReaderTracks.vue'
import KernelAtlas from '@theme/components/KernelAtlas.vue'
import SystemBlueprint from '@theme/components/SystemBlueprint.vue'
import ResearchLandscape from '@theme/components/ResearchLandscape.vue'
</script>

<WhitepaperHero />
<ReaderTracks />
<KernelAtlas />
<SystemBlueprint />

## What this library is actually shipping

Triton Fused Ops is a focused GPU kernel library for Transformer inference. It does **not** try to be a full model framework. It ships a small set of user-facing kernel families, a reference layer for verification, a validation layer for runtime contracts, and tooling for Benchmarking, Auto-Tuning, and Performance metrics.

<div class="link-grid link-grid-3">
  <a class="info-card" href="./overview/">
    <span class="card-kicker">Overview</span>
    <strong>Learn the vocabulary first</strong>
    <span>Start with the project terms, evidence model, and reading order used across the docs.</span>
  </a>
  <a class="info-card" href="./academy/">
    <span class="card-kicker">Academy</span>
    <strong>Take the interview-grade path</strong>
    <span>Read the system overview, then descend into kernel families and architecture notes.</span>
  </a>
  <a class="info-card" href="./guides/">
    <span class="card-kicker">Guides</span>
    <strong>Wire it into real code</strong>
    <span>Use the integration and performance guides when you are making deployment decisions.</span>
  </a>
</div>

## How to review claims in this repository

| Question | Where to look | What counts as evidence |
| :-- | :-- | :-- |
| What is the public promise? | `triton_ops.__init__`, kernel family pages, Architecture Lab | Exported launchers, wrappers, helpers, and their documented contracts |
| How is correctness checked? | Kernel family pages, `triton_ops.reference`, `triton_ops.validation`, `BenchmarkSuite` | Reference implementations, explicit validation, correctness verification |
| How are latency claims framed? | Benchmarking docs, benchmark suite, performance helpers | Warmup, synchronization, explicit problem shapes, derived metrics |
| Where does tuning stop? | `triton_ops.autotuner`, guides/performance | Auto-Tuning searches latency configs; it does not silently rewrite runtime semantics |

## Reading order for different jobs

1. **Evaluator** — read [Overview](/en/overview/), [Academy](/en/academy/), and [Architecture Lab](/en/architecture-lab/).
2. **Integrator** — go from [Kernel Families](/en/kernel-families/) to [Integration Guide](/en/guides/integration).
3. **Performance reviewer** — read [Performance Guide](/en/guides/performance), then [Reference & Research](/en/reference-research/).

<ResearchLandscape />
