---
title: 工程指南
---

# 工程指南

选择你需要的叙事路径。这些页面面向已经理解系统形状、现在需要做工程决策的读者。

<div class="link-grid link-grid-3">
  <a class="info-card" href="./integration">
    <span class="card-kicker">集成</span>
    <strong>把 Kernel family 接进模型代码</strong>
    <span>先画清运行时契约、模块封装与适配边界，再替换热点路径。</span>
  </a>
  <a class="info-card" href="./performance">
    <span class="card-kicker">性能</span>
    <strong>正确测量、调优并解释结果</strong>
    <span>把 Benchmarking、Auto-Tuning 与 Performance metrics 分开，才能让结论站得住。</span>
  </a>
  <a class="info-card" href="../reference-research/">
    <span class="card-kicker">研究</span>
    <strong>把实现放回更大的系统语境里</strong>
    <span>当你需要与相邻项目或论文做比较时，从研究部分进入。</span>
  </a>
</div>

## 选页启发式

- 当你要修改模型边界时，优先读 **integration**。
- 当你在审查 latency、Auto-Tuning 或 Performance metrics 时，优先读 **performance**。
- 当你需要外部比较视角再做架构判断时，转去 **reference-research**。
