---
layout: default
title: "Kernel Design — Triton Fused Ops"
description: "Triton kernel implementation details for Triton Fused Ops"
---

# Kernel Design

Technical deep dive into Triton kernel implementations.

---

## RMSNorm + RoPE Kernel

### Algorithm

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
    
    # Compute indices
    batch_idx = pid // seq_stride
    seq_idx = pid % seq_stride
    
    # Load input block
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    
    input_ptrs = input_ptr + batch_idx * batch_stride + seq_idx * seq_stride + offsets
    x = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Load weights
    weight_ptrs = weight_ptr + offsets
    w = tl.load(weight_ptrs, mask=mask, other=1.0)
    
    # RMSNorm: compute variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / hidden_dim
    rstd = tl.rsqrt(var + eps)
    
    # Normalize
    x_norm = x * rstd * w
    
    # RoPE: apply rotation
    # Load cos/sin
    cos_ptrs = cos_ptr + seq_idx * (hidden_dim // 2) + (offsets // 2)
    sin_ptrs = sin_ptr + seq_idx * (hidden_dim // 2) + (offsets // 2)
    cos_val = tl.load(cos_ptrs, mask=mask, other=1.0)
    sin_val = tl.load(sin_ptrs, mask=mask, other=0.0)
    
    # Rotate pairs
    x_rot = tl.where(offsets % 2 == 0, -x_norm[:, None], x_norm[:, None])
    x_rot = tl.reshape(x_rot, (BLOCK_SIZE,))
    
    # Apply RoPE
    x_rope = x_norm * cos_val + x_rot * sin_val
    
    # Store output
    output_ptrs = output_ptr + batch_idx * batch_stride + seq_idx * seq_stride + offsets
    tl.store(output_ptrs, x_rope, mask=mask)
```

### Memory Access Pattern

```
Standard (PyTorch):
┌─────────┐    HBM    ┌─────────┐    HBM    ┌─────────┐
│  Input  │ ───────►│ RMSNorm │ ───────►│  RoPE   │
│  (x)    │   read   │  Kernel │  write  │  Kernel │
└─────────┘          └─────────┘  read   └─────────┘
  └──────────────────────────────┘
       3 HBM reads, 2 HBM writes

Fused (Triton):
┌─────────┐                           ┌─────────┐
│  Input  │ ─────►┌─────────────┐────►│ Output  │
│  (x)    │  HBM  │ RMSNorm+RoPE│  HBM  │         │
└─────────┘       │  (SRAM)     │       └─────────┘
                   └─────────────┘
       1 HBM read, 1 HBM write
```

---

## Gated MLP Kernel

### Algorithm

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
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulators
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Compute gemm
    for k in range(0, in_features, BLOCK_K):
        # Load input block
        a_ptrs = input_ptr + offs_m[:, None] * batch_stride + offs_k[None, :]
        a = tl.load(a_ptrs, mask=offs_k[None, :] < in_features - k, other=0.0)
        
        # Load gate weight block
        gate_w_ptrs = gate_w_ptr + offs_k[:, None] * out_features + offs_n[None, :]
        gate_w = tl.load(gate_w_ptrs, mask=offs_n[None, :] < out_features, other=0.0)
        
        # Load up weight block
        up_w_ptrs = up_w_ptr + offs_k[:, None] * out_features + offs_n[None, :]
        up_w = tl.load(up_w_ptrs, mask=offs_n[None, :] < out_features, other=0.0)
        
        # Accumulate
        gate_acc += tl.dot(a, gate_w)
        up_acc += tl.dot(a, up_w)
        
        offs_k += BLOCK_K
    
    # Apply activation to gate
    if ACTIVATION == "silu":
        gate_acc = gate_acc * tl.sigmoid(gate_acc)
    elif ACTIVATION == "gelu":
        gate_acc = tl.gelu(gate_acc)
    
    # Element-wise multiply
    output = gate_acc * up_acc
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * out_features + offs_n[None, :]
    tl.store(output_ptrs, output, mask=(offs_m[:, None] < batch_stride) & (offs_n[None, :] < out_features))
```

---

## FP8 GEMM Kernel

### Algorithm

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
    # Map program ID to output block
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute offsets
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers to scales
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Main loop
    for k in range(0, K, BLOCK_K):
        # Load and dequantize A
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_fp8 = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        a_fp16 = a_fp8.to(tl.float16) / a_scale
        
        # Load and dequantize B
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_fp8 = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        b_fp16 = b_fp8.to(tl.float16) / b_scale
        
        # Compute dot product
        acc += tl.dot(a_fp16, b_fp16)
        
        offs_k += BLOCK_K
    
    # Write output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
```

---

## Tiling Strategy

### Block Size Selection

| Kernel | BLOCK_M | BLOCK_N | BLOCK_K | Considerations |
|:-------|:--------|:--------|:--------|:---------------|
| **GEMM** | 128-256 | 128-256 | 32-64 | Balance shared mem vs parallelism |
| **Element-wise** | 128-1024 | N/A | N/A | Match tensor dimensions |
| **Reduction** | 128-256 | N/A | N/A | Optimal for thread coarsening |

### Memory Hierarchy Usage

```
HBM (High Bandwidth Memory)
    │  ~1.5-2 TB/s
    ▼
L2 Cache
    │  ~10-20 TB/s
    ▼
Shared Memory (SRAM)
    │  ~10-20 TB/s
    ▼
Registers
    │  Fastest
    ▼
ALU
```

### Coalesced Memory Access

```python
# Good: Coalesced access
offsets = tl.arange(0, BLOCK_SIZE)
ptrs = base_ptr + offsets
values = tl.load(ptrs)  # Threads access consecutive addresses

# Bad: Strided access
offsets = tl.arange(0, BLOCK_SIZE) * stride
ptrs = base_ptr + offsets
values = tl.load(ptrs)  # Threads access strided addresses
```

---

<div align="center">

**[⬆ Back to Top](#kernel-design)** | **[← Back to Internals](../)**

</div>
