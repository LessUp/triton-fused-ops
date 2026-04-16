---
layout: default
title: "算子设计 — Triton Fused Ops"
description: "Triton Fused Ops Triton 算子实现细节"
---

# 算子设计

Triton 算子实现的技术深度解析。

---

## RMSNorm + RoPE 算子

### 算法

```python
@triton.jit
def rmsnorm_rope_kernel(
    input_ptr, weight_ptr, cos_ptr, sin_ptr, output_ptr,
    batch_stride, seq_stride, hidden_dim,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # 计算索引
    batch_idx = pid // seq_stride
    seq_idx = pid % seq_stride
    
    # 加载输入块
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    input_ptrs = input_ptr + batch_idx * batch_stride + seq_idx * seq_stride + offsets
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # 加载权重
    weight_ptrs = weight_ptr + offsets
    w = tl.load(weight_ptrs, mask=mask, other=1.0)
    
    # RMSNorm：计算方差
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / hidden_dim
    rstd = tl.rsqrt(var + eps)
    
    # 归一化
    x_norm = x * rstd * w
    
    # RoPE：应用旋转
    # 加载 cos/sin
    cos_ptrs = cos_ptr + seq_idx * (hidden_dim // 2) + (offsets // 2)
    sin_ptrs = sin_ptr + seq_idx * (hidden_dim // 2) + (offsets // 2)
    cos_val = tl.load(cos_ptrs, mask=mask, other=1.0)
    sin_val = tl.load(sin_ptrs, mask=mask, other=0.0)
    
    # 旋转对
    x_rot = tl.where(offsets % 2 == 0, -x_norm[:, None], x_norm[:, None])
    x_rot = tl.reshape(x_rot, (BLOCK_SIZE,))
    
    # 应用 RoPE
    x_rope = x_norm * cos_val + x_rot * sin_val
    
    # 存储输出
    output_ptrs = output_ptr + batch_idx * batch_stride + seq_idx * seq_stride + offsets
    tl.store(output_ptrs, x_rope, mask=mask)
```

### 内存访问模式

```
标准 (PyTorch)：
┌─────────┐        ┌─────────┐        ┌─────────┐
│  Input  │──HBM──►│ RMSNorm │──HBM──►│  RoPE   │
│  (x)    │  read  │  Kernel │ write  │  Kernel │
└─────────┘        └─────────┘  read  └─────────┘
  └──────────────────────────────┘
       3 次 HBM 读，2 次 HBM 写

融合 (Triton)：
┌─────────┐                           ┌─────────┐
│  Input  │─────────►┌─────────┐──────►│ Output  │
│  (x)    │    HBM   │ RMSNorm+│  HBM   │         │
└─────────┘   read   │  RoPE   │  write └─────────┘
                     │ (SRAM)  │
                     └─────────┘
                          │
                     寄存器/SRAM
                     (无 HBM 流量)

       1 次 HBM 读，1 次 HBM 写
```

---

## Gated MLP 算子

### 算法

```python
@triton.jit
def gated_mlp_kernel(
    input_ptr, gate_w_ptr, up_w_ptr, output_ptr,
    batch_stride, in_features, out_features,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 计算偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 计算 GEMM
    for k in range(0, in_features, BLOCK_K):
        # 加载输入块
        a_ptrs = input_ptr + offs_m[:, None] * batch_stride + offs_k[None, :]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < in_features - k, other=0.0)
        
        # 加载 gate 权重块
        gate_w_ptrs = gate_w_ptr + offs_k[:, None] * out_features + offs_n[None, :]
        gate_w = tl.load(gate_w_ptrs, mask=offs_n[None, :] < out_features, other=0.0)
        
        # 加载 up 权重块
        up_w_ptrs = up_w_ptr + offs_k[:, None] * out_features + offs_n[None, :]
        up_w = tl.load(up_w_ptrs, mask=offs_n[None, :] < out_features, other=0.0)
        
        # 累加
        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)
        
        offs_k += BLOCK_K
    
    # 对 gate 应用激活函数
    if ACTIVATION == "silu":
        gate_acc = gate_acc * tl.sigmoid(gate_acc)
    elif ACTIVATION == "gelu":
        gate_acc = tl.gelu(gate_acc)
    
    # 逐元素相乘
    output = gate_acc * up_acc
    
    # 存储结果
    output_ptrs = output_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < batch_stride) & (offs_n[None, :] < out_features))
```

---

## FP8 GEMM 算子

### 算法

```python
@triton.jit
def fp8_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # 将 Program ID 映射到输出块
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # 计算偏移
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 缩放指针
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
    
    # 初始化累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 主循环
    for k in range(0, K, BLOCK_K):
        # 加载并反量化 A
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        a_fp16 = a_fp8.to(tl.float16) / a_scale
        
        # 加载并反量化 B
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        b_fp16 = b_fp8.to(tl.float16) / b_scale
        
        # 计算点积
        acc += tl.dot(a_fp16, b_fp16)
        
        offs_k += BLOCK_K
    
    # 写入输出
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
```

---

## 分块策略

### 块大小选择

| 算子 | BLOCK_M | BLOCK_N | BLOCK_K | 考虑因素 |
|:-------|:--------|:--------|:--------|:---------------|
| **GEMM** | 128-256 | 128-256 | 32-64 | 平衡共享内存与并行性 |
| **逐元素** | 128-1024 | N/A | N/A | 匹配张量维度 |
| **归约** | 128-256 | N/A | N/A | 线程粗化的最优选择 |

### 内存层次结构使用

```
HBM (高带宽内存)
    │  ~1.5-2 TB/s
    ▼
L2 缓存
    │  ~10-20 TB/s
    ▼
共享内存 (SRAM)
    │  ~10-20 TB/s
    ▼
寄存器
    │  最快
    ▼
ALU
```

### 合并内存访问

```python
# 良好：合并访问
offsets = tl.arange(0, BLOCK_SIZE)
ptrs = base_ptr + offsets
values = tl.load(ptrs)  # 线程访问连续地址

# 不好：步进访问
offsets = tl.arange(0, BLOCK_SIZE) * stride
ptrs = base_ptr + offsets
values = tl.load(ptrs)  # 线程访问步进地址
```

---

<div align="center">

**[⬆ 返回顶部](#算子设计)** | **[← 返回内部文档](../)**

</div>
