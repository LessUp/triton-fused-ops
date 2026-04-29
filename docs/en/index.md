---
layout: default
title: Documentation
nav_title: English
nav_order: 2
has_children: true
permalink: /docs/en/
---

# English Knowledge Base

This section is a code-accurate reference for the current repository state: public APIs, runtime contracts, performance tooling, and kernel internals.

<div class="link-grid link-grid-2">
  <a class="info-card" href="{{ '/docs/en/getting-started/' | relative_url }}">
    <span class="card-kicker">Getting Started</span>
    <strong>Install, run, and copy working snippets</strong>
    <span>Start from environment setup, the first fused calls, and module-wrapper examples.</span>
  </a>
  <a class="info-card" href="{{ '/docs/en/api/' | relative_url }}">
    <span class="card-kicker">API Reference</span>
    <strong>Public surface and contracts</strong>
    <span>Kernel signatures, quantization helpers, autotuning, benchmark classes, models, and errors.</span>
  </a>
  <a class="info-card" href="{{ '/docs/en/guides/' | relative_url }}">
    <span class="card-kicker">Guides</span>
    <strong>Integration and performance knowledge</strong>
    <span>Where to place fused ops, how to measure them, and how to use FP8 responsibly.</span>
  </a>
  <a class="info-card" href="{{ '/docs/en/internals/' | relative_url }}">
    <span class="card-kicker">Internals</span>
    <strong>Source-level implementation context</strong>
    <span>Architecture, kernel design trade-offs, and memory-traffic reduction patterns.</span>
  </a>
</div>

## Reading paths

<div class="callout-grid">
  <div class="note-panel">
    <strong>First visit</strong>
    <p>Read <a href="{{ '/docs/en/getting-started/installation/' | relative_url }}">Installation</a> and <a href="{{ '/docs/en/getting-started/quickstart/' | relative_url }}">Quick Start</a>.</p>
  </div>
  <div class="note-panel">
    <strong>API integration</strong>
    <p>Read <a href="{{ '/docs/en/api/kernels/' | relative_url }}">Core Kernels</a> and <a href="{{ '/docs/en/guides/integration/' | relative_url }}">Integration</a>.</p>
  </div>
  <div class="note-panel">
    <strong>Performance work</strong>
    <p>Read <a href="{{ '/docs/en/api/benchmark/' | relative_url }}">Benchmark</a>, <a href="{{ '/docs/en/api/autotuner/' | relative_url }}">Auto-Tuning</a>, and <a href="{{ '/docs/en/guides/performance/' | relative_url }}">Performance</a>.</p>
  </div>
  <div class="note-panel">
    <strong>Source dive</strong>
    <p>Read <a href="{{ '/docs/en/internals/architecture/' | relative_url }}">Architecture</a> and <a href="{{ '/docs/en/internals/kernel-design/' | relative_url }}">Kernel Design</a>.</p>
  </div>
</div>

## Boundary reminder

- Triton kernel execution requires CUDA.
- CPU-only environments remain useful for import checks, linting, typing, build validation, and CPU-safe tests.
- The site intentionally keeps only technical knowledge pages; repository process history and changelog content are not part of the published docs.
