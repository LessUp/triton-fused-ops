---
title: 异常模型
description: "异常层级与附带元数据说明"
---

# 异常模型

本仓库所有自定义异常都继承自 `TritonKernelError`。

## 层级

```text
TritonKernelError
├── ShapeMismatchError
├── UnsupportedDtypeError
├── NumericalOverflowError
├── TuningFailedError
└── DeviceError
```

## `ShapeMismatchError`

可能附带：

- `expected`
- `actual`
- `tensor_name`

典型场景：kernel 输入的 tensor 维度不兼容。

## `UnsupportedDtypeError`

可能附带：

- `dtype`
- `supported_dtypes`
- `tensor_name`

典型场景：把整数 tensor 传入仅支持浮点的 kernel 路径。

## `NumericalOverflowError`

可能附带：

- `max_value`
- `scale`
- `attempts`

典型场景：FP8 量化多次缩小 scale 后仍然超出范围。

## `TuningFailedError`

可能附带：

- `problem_size`
- `configs_tried`
- `last_error`

典型场景：自动调优时所有候选配置都执行失败。

## `DeviceError`

可能附带：

- `expected_device`
- `actual_device`
- `tensor_name`

典型场景：用 CPU tensor 调用 Triton kernel，或输入分布在不同设备上。
