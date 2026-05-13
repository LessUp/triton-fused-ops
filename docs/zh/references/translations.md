---
title: 术语翻译对照
description: "本文档使用的技术术语中英对照表"
---

# 术语翻译对照

本页汇总文档中使用的核心术语及其标准中英对照，方便跨语言阅读和引用。

## 通用架构术语

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| Kernel | 算子 / 内核 | Triton 编译后在 GPU 上执行的函数 |
| Fusion | 融合 | 将多个计算步骤合并为单次 GPU 启动 |
| Fused operator | 融合算子 | 单次 kernel 完成多步计算的结果 |
| Launcher | 启动器 | Python 端负责调用 Triton kernel 的包装函数 |
| Reference implementation | 参考实现 | 用于验证正确性的纯 CPU/NumPy 版本 |
| Module wrapper | 模块封装 | `torch.nn.Module` 形式的包装接口 |

## 内存与性能术语

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| HBM (High Bandwidth Memory) | 高带宽显存 | GPU 全局显存 |
| SRAM / Shared Memory | 共享内存 | GPU 片上高速缓存，容量小但带宽极高 |
| Register | 寄存器 | GPU 线程私有存储，速度最快 |
| Memory-bound | 内存受限 | 性能瓶颈在显存带宽而非算力 |
| Compute-bound | 计算受限 | 性能瓶颈在 ALU 算力而非显存带宽 |
| Memory traffic | 内存流量 | 一次前向/反向过程中读写显存的总量 |
| Cache locality | 缓存局部性 | 数据在时间和空间上的复用程度 |
| Coalescing | 合并访问 | 相邻线程访问相邻内存地址的优化模式 |

## Triton 专用术语

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| Block pointer | 块指针 | Triton 中用于指定块级内存加载的抽象 |
| Tile / Tiling | 分块 | 将大矩阵/张量划分为适合 SRAM 的小块 |
| Block size | 块大小 | 每个 Triton program 实例处理的数据维度 |
| Program ID | 程序 ID | Triton 中用于索引当前块的坐标 |
| Mask | 掩码 | 处理边界条件时屏蔽越界线程的布尔数组 |
| Autotuning | 自动调优 | 搜索最优 block size、num_stages 等启动参数 |
| Config space | 配置空间 | 所有待搜索的启动参数组合 |

## 量化术语

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| Quantization | 量化 | 将高精度浮点数映射到低精度整数表示 |
| Dequantization | 反量化 | 将量化值还原为浮点近似值 |
| Scale | 缩放因子 | 量化/反量化时使用的 per-tensor 或 per-channel 乘数 |
| E4M3 / E5M2 | — | FP8 的两种 IEEE 754 变体格式 |
| Overflow handling | 溢出处理 | 当值超出量化表示范围时的回退策略 |
| Per-tensor | 逐张量 | 对整个张量使用单一 scale |
| Per-channel | 逐通道 | 对张量的每个通道使用独立 scale |

## 测试与度量术语

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| Warmup | 预热 | 正式计时前空跑若干轮以稳定 GPU 状态 |
| Benchmark suite | 基准测试套件 | 包含正确性验证 + 性能测量 + 报告生成的完整工具 |
| Correctness verification | 正确性验证 | 与参考实现对比输出误差是否在容差内 |
| Latency | 延迟 | 单次 kernel 执行的 wall-clock 时间 |
| Throughput | 吞吐 | 单位时间内处理的数据量 |
| Speedup | 加速比 | 相比基准实现的耗时倍数提升 |
| Synchronize | 同步 | `torch.cuda.synchronize()`，强制等待 GPU 完成所有队列任务 |

## 文档相关

| 英文 | 中文 | 说明 |
|:--|:--|:--|
| Frontmatter | 页首元数据 | Markdown 文件顶部的 YAML 配置块 |
| Layout | 布局 | 页面渲染模板（如 `home`、`doc`） |
| Sidebar | 侧边栏 | 页面左侧的章节导航 |
| Nav / Navigation | 导航栏 | 页面顶部的全局导航 |
