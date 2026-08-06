# Triton Fused Ops 路线图

> 定位：Triton kernel 与验证方法的**精简练习仓**，当前处于维护模式。
> 三条算子路径（fused_rmsnorm_rope / fused_gated_mlp / flash_attention）已具备
> 独立参考实现与差分测试，这是本仓库的核心资产。

## 面试前建议（低成本）

- [ ] 在可用 GPU 上跑一次完整 benchmark，把真实数字（带硬件型号）写入 README
      —— 基准与 autotuner 基础设施已存在，只缺一次真实执行
- [ ] 准备「Triton 版 FlashAttention vs CUDA C++ 版（cuflash-attn）」的对比讲述：
      两者的 block 设计、在线 softmax 实现与验证方法异同

## 可选扩展（只在有余力时）

- [ ] 新增融合算子候选：fused softmax + mask、INT8/FP8 反量化融合（必须带独立参考实现与差分测试才进主分支）
- [ ] 与 tiny-llm 的 kernel 选型呼应：说明何时选 Triton、何时选 CUDA C++

## 明确不做

- 不引入无参考实现的 kernel
- 不写入未在真实硬件测量的性能数字（沿用仓库既有原则）
