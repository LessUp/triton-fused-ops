---
layout: default
title: Guides
parent: Documentation
nav_order: 3
has_children: true
permalink: /docs/en/guides/
---

# Guides

These pages focus on engineering decisions: where to integrate the kernels, how to measure them, and how to reason about FP8 trade-offs.

<div class="link-grid link-grid-3">
  <a class="info-card" href="{{ '/docs/en/guides/integration/' | relative_url }}">
    <span class="card-kicker">Integration</span>
    <strong>Runtime contracts and model boundaries</strong>
    <span>Choose between the functional API, module wrappers, and custom adapters.</span>
  </a>
  <a class="info-card" href="{{ '/docs/en/guides/performance/' | relative_url }}">
    <span class="card-kicker">Performance</span>
    <strong>Measurement and tuning</strong>
    <span>Benchmark correctly, interpret metrics, and tune custom kernels.</span>
  </a>
  <a class="info-card" href="{{ '/docs/en/guides/fp8-best-practices/' | relative_url }}">
    <span class="card-kicker">FP8</span>
    <strong>Quantization best practices</strong>
    <span>Apply FP8 where it helps and keep numerically sensitive steps in higher precision.</span>
  </a>
</div>
