---
layout: default
title: Errors
parent: API Reference
grand_parent: Documentation
nav_order: 7
description: "Exception hierarchy and attached metadata"
---

# Errors

All custom exceptions in this repository inherit from `TritonKernelError`.

## Hierarchy

```text
TritonKernelError
├── ShapeMismatchError
├── UnsupportedDtypeError
├── NumericalOverflowError
├── TuningFailedError
└── DeviceError
```

## `ShapeMismatchError`

Attached attributes may include:

- `expected`
- `actual`
- `tensor_name`

Typical use: incompatible tensor dimensions for kernel inputs.

## `UnsupportedDtypeError`

Attached attributes may include:

- `dtype`
- `supported_dtypes`
- `tensor_name`

Typical use: integer tensor passed into a floating-only kernel path.

## `NumericalOverflowError`

Attached attributes may include:

- `max_value`
- `scale`
- `attempts`

Typical use: FP8 quantization still overflows after repeated scale reduction.

## `TuningFailedError`

Attached attributes may include:

- `problem_size`
- `configs_tried`
- `last_error`

Typical use: every candidate configuration fails during autotuning.

## `DeviceError`

Attached attributes may include:

- `expected_device`
- `actual_device`
- `tensor_name`

Typical use: calling a Triton kernel path with CPU tensors or mixed-device inputs.

## Catching patterns

```python
from triton_ops import fused_rmsnorm_rope
from triton_ops import DeviceError, ShapeMismatchError, UnsupportedDtypeError

try:
    y = fused_rmsnorm_rope(x, weight, cos, sin)
except DeviceError as exc:
    print(exc.expected_device, exc.actual_device)
except ShapeMismatchError as exc:
    print(exc.tensor_name, exc.expected, exc.actual)
except UnsupportedDtypeError as exc:
    print(exc.tensor_name, exc.dtype)
```
