---
layout: default
title: "内存优化 — Triton Fused Ops"
description: "Triton Fused Ops 融合策略和内存优化技术"
---

# 内存优化

融合策略和内存优化技术。

---

## 融合策略

### 为什么要融合？

| 操作 | 未融合 | 融合 | 改进 |
|:----------|:--------|:------|:------------|
| **RMSNorm + RoPE** | 5 次 HBM 访问 | 2 次 HBM 访问 | **减少 2.5 倍** |
| **Gated MLP** | 4 次 HBM 访问 | 2 次 HBM 访问 | **减少 2 倍** |
| **FP8 量化 + GEMM** | 分离操作 | 单 kernel | **加速 1.5 倍** |

### 内存带宽瓶颈

```
标准实现：
┌─────────┐        ┌─────────┐        ┌─────────┐
│  Input  │──HBM──►│ Kernel1 │──HBM──►│ Kernel2 │──HBM──► Output
│  (HBM)  │  read  │  (HBM)  │ write  │  (HBM)  │ write │  (HBM)
└─────────┘        └─────────┘  read  └─────────┘       └─────────┘
                                                              
总计：每个元素 3 次读，2 次写
峰值带宽：~30-40% 利用率

融合实现：
┌─────────┐                             ┌─────────┐
│  Input  │─────────►┌─────────┐────────►│ Output  │
│  (HBM)  │    HBM   │  Fused  │   HBM   │  (HBM)  │
└─────────┘   read   │  Kernel │  write  └─────────┘
                     │ (SRAM)  │
                     └─────────┘
                          │
                     寄存器/SRAM
                     (无 HBM 流量)

总计：每个元素 1 次读，1 次写
峰值带宽：90%+ 利用率
```

---

## SRAM 使用

### 寄存器压力

| 算子 | 每线程寄存器 | 每块线程数 | 总寄存器数 |
|:-------|:--------------------|:------------------|:----------------|
| **RMSNorm** | 64-128 | 128-256 | 8K-32K |
| **GEMM** | 128-256 | 128-256 | 16K-64K |
| **归约** | 32-64 | 256-512 | 8K-32K |

### 共享内存布局

```python
@triton.jit
def optimized_kernel(...):
    # 分配共享内存
    smem = tl.static_alloc_shared(4 * 1024)  # 每块 4KB
    
    # 加载到 SMEM
    smem_ptr = smem + tl.arange(0, BLOCK_SIZE)
    tl.store(smem_ptr, values)
    tl.debug_barrier()  # 确保所有线程已存储
    
    # 从 SMEM 读取
    values = tl.load(smem_ptr)
```

---

## 量化影响

### FP8 内存节省

| 组件 | FP16 | FP8 | 节省 |
|:----------|:----:|:---:|:--------|
| **权重** | 100% | 50% | **50%** |
| **激活** | 100% | 50% | **50%** |
| **KV 缓存** | 100% | 50% | **50%** |

### FP8 内存带宽

```
FP16 GEMM：
┌─────────┐     2 字节     ┌─────────┐
│   A     │───────────────►│  ALU    │
│ (FP16)  │     per elem   │         │
└─────────┘                └─────────┘
读取带宽: 2 * M * K 字节

FP8 GEMM：
┌─────────┐     1 字节     ┌─────────┐
│   A     │──────────────►│  ALU    │
│  (FP8)  │    per elem   │         │
└─────────┘               └─────────┘
读取带宽: 1 * M * K 字节
等效带宽: 2 倍
```

---

## 缓存优化

### L2 缓存使用

```python
@triton.jit
def cache_optimized_kernel(
    input_ptr, output_ptr,
    stride_m, stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # 分组块以获得更好的 L2 局部性
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # 混淆 program IDs
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    # 计算内存偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # 访问模式
    ptrs = input_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
```

### 内存合并模式

```
良好（合并）：
Thread 0: addr + 0
Thread 1: addr + 1
Thread 2: addr + 2
Thread 3: addr + 3
...
→ 单次内存事务

不好（未合并）：
Thread 0: addr + 0
Thread 1: addr + 64
Thread 2: addr + 128
Thread 3: addr + 192
...
→ 多次内存事务
```

---

## 优化检查清单

### 算子设计

- [ ] 最小化 HBM 访问
- [ ] 最大化寄存器/SRAM 使用
- [ ] 使用合并内存访问
- [ ] 避免共享内存中的 bank 冲突
- [ ] 平衡并行性与资源使用

### 融合机会

- [ ] Matmul 前后的逐元素操作
- [ ] 归一化 + 激活函数
- [ ] 量化 + 反量化对
- [ ] 多个小的归约操作

### 自动调优

- [ ] 尝试不同的块大小
- [ ] 调整 warp 数量
- [ ] 内存操作的流水线级数
- [ ] 缓存配置

---

<div align="center">

**[⬆ 返回顶部](#内存优化)** | **[← 返回内部文档](../)**

</div>
