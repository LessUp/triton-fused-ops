---
layout: default
title: Validation
parent: API Reference
grand_parent: Documentation
nav_order: 6
description: "Validation helpers and the runtime contracts they enforce"
---

# Validation

The validation helpers live in `triton_ops.validation`. They are useful when building wrappers, tests, or custom entry points around the shipped kernels.

## Core helpers

### `validate_rmsnorm_rope_inputs`

Checks:

- CUDA placement for `x`, `weight`, `cos`, and `sin`
- same-device requirement across all inputs
- supported floating dtypes
- contiguity
- `x` is 3D
- `weight.shape == (hidden_dim,)`
- `cos` is 2D or 4D
- `sin.shape == cos.shape`
- `hidden_dim % head_dim == 0` when `num_heads` is inferred

Return value:

```python
(batch_size, seq_len, hidden_dim, head_dim, num_heads)
```

### `validate_gated_mlp_inputs`

Checks:

- CUDA placement and same-device requirement
- supported floating dtypes
- contiguity
- `x` is 3D
- both weight tensors are 2D and have matching shapes
- activation is `"silu"` or `"gelu"`

Return value:

```python
(batch_size, seq_len, hidden_dim, intermediate_dim)
```

### `validate_fp8_gemm_inputs`

Checks:

- CUDA placement for matrices and provided scales
- contiguity for `a` and `b`
- 2D matrix shapes
- matching inner dimension `K`
- required scale tensors when pre-quantized FP8 inputs are used
- accepted output dtype according to the validation layer

Return value:

```python
(M, N, K)
```

### `validate_fp8_quantize_inputs`

Checks:

- CUDA placement
- floating dtype support
- contiguity
- scalar positive `scale` when provided

## Scalar checks

Available helpers:

- `validate_positive_dimensions(**dims)`
- `validate_head_dim(head_dim)`
- `validate_eps(eps)`

These are used internally by the kernel entry points and are also useful when you build higher-level wrappers that want to fail early before launching Triton.
