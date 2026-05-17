---
title: Related Projects
description: Curated commentary on open-source projects adjacent to Triton Fused Ops
---

# Related Projects

These projects matter because they define the ecosystem expectations around kernel authoring, tensor runtime behavior, or inference deployment. The commentary below is intentionally narrow: what does each project help us compare or reason about?

## Core foundations

| Project | Why it matters here |
| :-- | :-- |
| OpenAI Triton | The compiler and Python DSL beneath every Triton kernel in this repo. It defines the implementation medium, not just a build dependency. |
| PyTorch | The tensor runtime for the public API, module wrappers, and most validation assumptions. It anchors the user contract. |

## Inference and systems neighbors

| Project | Curated commentary |
| :-- | :-- |
| vLLM | Useful as a deployment-context reference. The repo does not ship a vLLM adapter today, but vLLM helps frame what real serving systems expect from optimized primitives. |
| TensorRT-LLM | A strong comparison point for industrial inference stacks, especially when asking what parts of a production stack live outside a focused kernel library. |
| xFormers | Helpful for understanding how another project packages efficient Transformer building blocks without claiming to replace the whole framework stack. |
| CUTLASS | Important when thinking about matrix-multiplication structure, tile ordering, and how lower-level kernel patterns influence FP8 GEMM design. |

## Practical reading rule

Use related projects to sharpen questions, not to borrow prestige. The useful question is never “does this repo mention the big names?” It is “which adjacent project clarifies the trade-offs of this specific design decision?”
