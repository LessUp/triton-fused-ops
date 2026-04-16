---
layout: default
title: "Memory Optimization — Triton Fused Ops"
description: "Fusion strategies and memory optimization techniques"
---

# Memory Optimization

Fusion strategies and memory optimization techniques.

---

## Fusion Strategy

### Why Fuse?

| Operation | Unfused | Fused | Improvement |
|:----------|:--------|:------|:------------|
| **RMSNorm + RoPE** | 5 HBM accesses | 2 HBM accesses | **2.5x** reduction |
| **Gated MLP** | 4 HBM accesses | 2 HBM accesses | **2x** reduction |
| **FP8 Quant + GEMM** | Separate ops | Single kernel | **1.5x** speedup |

### Memory Bandwidth Bottleneck

```
Standard Implementation:
┌─────────┐        ┌─────────┐        ┌─────────┐
│  Input  │──HBM──►│ Kernel1 │──HBM──►│ Kernel2 │──HBM──► Output
│  (HBM)  │  read  │  (HBM)  │ write  │  (HBM)  │ write │  (HBM)
└─────────┘        └─────────┘  read  └─────────┘       └─────────┘
                                                              
Total: 3 reads, 2 writes per element
Peak bandwidth: ~30-40% utilized

Fused Implementation:
┌─────────┐                             ┌─────────┐
│  Input  │─────────►┌─────────┐────────►│ Output  │
│  (HBM)  │    HBM   │  Fused  │   HBM   │  (HBM)  │
└─────────┘   read   │  Kernel │  write  └─────────┘
                     │ (SRAM)  │
                     └─────────┘
                          │
                     Registers/SRAM
                     (No HBM traffic)

Total: 1 read, 1 write per element
Peak bandwidth: 90%+ utilized
```

---

## SRAM Utilization

### Register Pressure

| Kernel | Registers per Thread | Threads per Block | Total Registers |
|:-------|:--------------------|:------------------|:----------------|
| **RMSNorm** | 64-128 | 128-256 | 8K-32K |
| **GEMM** | 128-256 | 128-256 | 16K-64K |
| **Reduction** | 32-64 | 256-512 | 8K-32K |

### Shared Memory Layout

```python
@triton.jit
def optimized_kernel(...):
    # Allocate shared memory
    smem = tl.static_alloc_shared(4 * 1024)  # 4KB per block
    
    # Load to SMEM
    smem_ptr = smem + tl.arange(0, BLOCK_SIZE)
    tl.store(smem_ptr, values)
    tl.debug_barrier()  # Ensure all threads stored
    
    # Read from SMEM
    values = tl.load(smem_ptr)
```

---

## Quantization Impact

### FP8 Memory Savings

| Component | FP16 | FP8 | Savings |
|:----------|:----:|:---:|:--------|
| **Weights** | 100% | 50% | **50%** |
| **Activations** | 100% | 50% | **50%** |
| **KV Cache** | 100% | 50% | **50%** |

### Memory Bandwidth with FP8

```
FP16 GEMM:
┌─────────┐     2 bytes     ┌─────────┐
│   A     │───────────────►│  ALU    │
│ (FP16)  │     per elem   │         │
└─────────┘                └─────────┘
Read bandwidth: 2 * M * K bytes

FP8 GEMM:
┌─────────┐     1 byte     ┌─────────┐
│   A     │──────────────►│  ALU    │
│  (FP8)  │    per elem   │         │
└─────────┘               └─────────┘
Read bandwidth: 1 * M * K bytes
Effective bandwidth: 2x
```

---

## Cache Optimization

### L2 Cache Utilization

```python
@triton.jit
def cache_optimized_kernel(
    input_ptr, output_ptr,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Group blocks for better L2 locality
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Swizzle program IDs
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # Compute memory offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Access pattern
    ptrs = input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
```

### Memory Coalescing Patterns

```
Good (Coalesced):
Thread 0: addr + 0
Thread 1: addr + 1
Thread 2: addr + 2
Thread 3: addr + 3
...
→ Single memory transaction

Bad (Uncoalesced):
Thread 0: addr + 0
Thread 1: addr + 64
Thread 2: addr + 128
Thread 3: addr + 192
...
→ Multiple memory transactions
```

---

## Optimization Checklist

### Kernel Design

- [ ] Minimize HBM accesses
- [ ] Maximize register/SRAM usage
- [ ] Use coalesced memory access
- [ ] Avoid bank conflicts in shared memory
- [ ] Balance parallelism vs resource usage

### Fusion Opportunities

- [ ] Element-wise ops before/after matmul
- [ ] Normalization + activation
- [ ] Quantization + dequantization pairs
- [ ] Multiple small reductions

### Auto-Tuning

- [ ] Try different block sizes
- [ ] Adjust warp counts
- [ ] Pipeline stages for memory operations
- [ ] Cache configurations

---

<div align="center">

**[⬆ Back to Top](#memory-optimization)** | **[← Back to Internals](../)**

</div>
