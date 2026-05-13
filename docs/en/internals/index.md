---
title: Internals
---

# Internals

This section explains how the library is structured and why the kernels are organized the way they are.

<div class="link-grid link-grid-3">
  <a class="info-card" href="/en/internals/architecture">
    <span class="card-kicker">Architecture</span>
    <strong>Module layout and responsibilities</strong>
    <span>See how the public API, validation helpers, kernels, autotuner, and benchmark code fit together.</span>
  </a>
  <a class="info-card" href="/en/internals/kernel-design">
    <span class="card-kicker">Kernel Design</span>
    <strong>Tiling and fusion strategy</strong>
    <span>Read the core ideas behind the Triton kernels and their memory access patterns.</span>
  </a>
  <a class="info-card" href="/en/internals/memory-optimization">
    <span class="card-kicker">Memory</span>
    <strong>HBM reduction and SRAM reuse</strong>
    <span>Understand why fusion matters and how the library reduces traffic to global memory.</span>
  </a>
</div>
