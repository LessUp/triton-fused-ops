---
title: Benchmark Visualization
description: "Visual performance comparisons of Triton Fused Ops kernels"
---

# Benchmark Visualization

These charts illustrate representative performance trends measured on the repository kernels. The numbers are directional references from an NVIDIA A100 SXM4 80GB environment.

## Latency vs Sequence Length (`fused_rmsnorm_rope`)

<div style="background: var(--vp-c-bg-soft); border: 1px solid var(--vp-c-border); border-radius: 12px; padding: 24px; margin: 24px 0;">
  <div style="display: flex; justify-content: space-between; align-items: flex-end; height: 280px; gap: 16px; padding-bottom: 40px; border-bottom: 1px solid var(--vp-c-border); position: relative;">
    <!-- Y-axis labels -->
    <div style="position: absolute; left: -40px; top: 0; bottom: 40px; display: flex; flex-direction: column; justify-content: space-between; font-size: 11px; color: var(--vp-c-text-3);">
      <span>80&micro;s</span>
      <span>60</span>
      <span>40</span>
      <span>20</span>
      <span>0</span>
    </div>

    <!-- 128 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 18%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~14&micro;s"></div>
        <div style="width: 24px; height: 40%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~32&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">128</span>
    </div>

    <!-- 512 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 25%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~20&micro;s"></div>
        <div style="width: 24px; height: 55%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~44&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">512</span>
    </div>

    <!-- 1024 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 35%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~28&micro;s"></div>
        <div style="width: 24px; height: 85%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~68&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">1024</span>
    </div>

    <!-- 2048 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 48%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~38&micro;s"></div>
        <div style="width: 24px; height: 100%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~80&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">2048</span>
    </div>

    <!-- 4096 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 62%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~50&micro;s"></div>
        <div style="width: 24px; height: 100%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~80&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">4096</span>
    </div>

    <!-- 8192 -->
    <div style="flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="display: flex; gap: 4px; align-items: flex-end; height: 220px;">
        <div style="width: 24px; height: 75%; background: #76B900; border-radius: 4px 4px 0 0;" title="Fused: ~60&micro;s"></div>
        <div style="width: 24px; height: 100%; background: #30363d; border-radius: 4px 4px 0 0;" title="Unfused: ~80&micro;s"></div>
      </div>
      <span style="font-size: 12px; color: var(--vp-c-text-3);">8192</span>
    </div>
  </div>

  <!-- Legend -->
  <div style="display: flex; justify-content: center; gap: 24px; margin-top: 16px; font-size: 13px;">
    <div style="display: flex; align-items: center; gap: 6px;">
      <div style="width: 14px; height: 14px; background: #76B900; border-radius: 3px;"></div>
      <span style="color: var(--vp-c-text-2);">Fused (this repo)</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px;">
      <div style="width: 14px; height: 14px; background: #30363d; border-radius: 3px;"></div>
      <span style="color: var(--vp-c-text-2);">Unfused (PyTorch)</span>
    </div>
  </div>
  <p style="text-align: center; font-size: 12px; color: var(--vp-c-text-3); margin-top: 8px;">Sequence Length &mdash; Latency (&micro;s) at batch=2, hidden_dim=4096</p>
</div>

## Speedup vs Batch Size

<div style="background: var(--vp-c-bg-soft); border: 1px solid var(--vp-c-border); border-radius: 12px; padding: 24px; margin: 24px 0;">
  <svg viewBox="0 0 600 240" style="width: 100%; height: auto;">
    <!-- Grid lines -->
    <line x1="50" y1="20" x2="50" y2="200" stroke="#30363d" stroke-width="1"/>
    <line x1="50" y1="200" x2="580" y2="200" stroke="#30363d" stroke-width="1"/>
    <line x1="50" y1="155" x2="580" y2="155" stroke="#21262d" stroke-width="1" stroke-dasharray="4"/>
    <line x1="50" y1="110" x2="580" y2="110" stroke="#21262d" stroke-width="1" stroke-dasharray="4"/>
    <line x1="50" y1="65" x2="580" y2="65" stroke="#21262d" stroke-width="1" stroke-dasharray="4"/>

    <!-- Y-axis labels -->
    <text x="40" y="25" text-anchor="end" fill="#6e7681" font-size="11">3.0&times;</text>
    <text x="40" y="70" text-anchor="end" fill="#6e7681" font-size="11">2.3</text>
    <text x="40" y="115" text-anchor="end" fill="#6e7681" font-size="11">1.7</text>
    <text x="40" y="160" text-anchor="end" fill="#6e7681" font-size="11">1.0</text>
    <text x="40" y="205" text-anchor="end" fill="#6e7681" font-size="11">0.3</text>

    <!-- X-axis labels -->
    <text x="110" y="220" text-anchor="middle" fill="#6e7681" font-size="11">1</text>
    <text x="230" y="220" text-anchor="middle" fill="#6e7681" font-size="11">4</text>
    <text x="350" y="220" text-anchor="middle" fill="#6e7681" font-size="11">16</text>
    <text x="470" y="220" text-anchor="middle" fill="#6e7681" font-size="11">64</text>
    <text x="550" y="220" text-anchor="middle" fill="#6e7681" font-size="11">128</text>

    <!-- Area fill -->
    <polygon points="50,185 110,160 230,130 350,95 470,70 550,65 550,200 50,200" fill="rgba(118,185,0,0.08)" stroke="none"/>

    <!-- Line -->
    <polyline points="50,185 110,160 230,130 350,95 470,70 550,65" fill="none" stroke="#76B900" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>

    <!-- Points -->
    <circle cx="50" cy="185" r="4" fill="#76B900"/>
    <circle cx="110" cy="160" r="4" fill="#76B900"/>
    <circle cx="230" cy="130" r="4" fill="#76B900"/>
    <circle cx="350" cy="95" r="4" fill="#76B900"/>
    <circle cx="470" cy="70" r="4" fill="#76B900"/>
    <circle cx="550" cy="65" r="4" fill="#76B900"/>

    <!-- Baseline -->
    <line x1="50" y1="155" x2="580" y2="155" stroke="#ff5454" stroke-width="1" stroke-dasharray="6,4"/>
    <text x="560" y="150" fill="#ff5454" font-size="10">1.0&times; baseline</text>
  </svg>
  <p style="text-align: center; font-size: 12px; color: var(--vp-c-text-3); margin-top: 8px;">Batch Size &mdash; Speedup Ratio (`fused_rmsnorm_rope`, seq_len=2048)</p>
</div>

## Memory Traffic Breakdown

<div style="background: var(--vp-c-bg-soft); border: 1px solid var(--vp-c-border); border-radius: 12px; padding: 24px; margin: 24px 0;">
  <div style="display: flex; justify-content: center; gap: 60px; align-items: flex-end; height: 240px; padding-bottom: 40px; border-bottom: 1px solid var(--vp-c-border);">

    <!-- Fused -->
    <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="width: 60px; display: flex; flex-direction: column; align-items: center;">
        <div style="width: 100%; height: 60px; background: linear-gradient(180deg, #3476f6, #1a4a9e); border-radius: 4px 4px 0 0; display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 10px; color: #fff; font-weight: 600;">Read</span>
        </div>
        <div style="width: 100%; height: 20px; background: linear-gradient(180deg, #ffc517, #c49000); display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 10px; color: #1a1a1a; font-weight: 600;">Write</span>
        </div>
        <div style="width: 100%; height: 80px; background: linear-gradient(180deg, #76B900, #5a8a00); border-radius: 0 0 4px 4px; display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 10px; color: #0d1117; font-weight: 600;">Reg</span>
        </div>
      </div>
      <span style="font-size: 13px; color: var(--vp-c-text-2); font-weight: 600; margin-top: 8px;">Fused</span>
    </div>

    <!-- Unfused -->
    <div style="display: flex; flex-direction: column; align-items: center; gap: 4px;">
      <div style="width: 60px; display: flex; flex-direction: column; align-items: center;">
        <div style="width: 100%; height: 100px; background: linear-gradient(180deg, #3476f6, #1a4a9e); border-radius: 4px 4px 0 0; display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 10px; color: #fff; font-weight: 600;">Read</span>
        </div>
        <div style="width: 100%; height: 60px; background: linear-gradient(180deg, #ffc517, #c49000); display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 10px; color: #1a1a1a; font-weight: 600;">Write</span>
        </div>
        <div style="width: 100%; height: 10px; background: linear-gradient(180deg, #76B900, #5a8a00); border-radius: 0 0 4px 4px; display: flex; align-items: center; justify-content: center;">
          <span style="font-size: 8px; color: #0d1117; font-weight: 600;">Reg</span>
        </div>
      </div>
      <span style="font-size: 13px; color: var(--vp-c-text-2); font-weight: 600; margin-top: 8px;">Unfused</span>
    </div>

  </div>

  <!-- Legend -->
  <div style="display: flex; justify-content: center; gap: 24px; margin-top: 16px; font-size: 13px;">
    <div style="display: flex; align-items: center; gap: 6px;">
      <div style="width: 14px; height: 14px; background: linear-gradient(180deg, #3476f6, #1a4a9e); border-radius: 3px;"></div>
      <span style="color: var(--vp-c-text-2);">HBM Read</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px;">
      <div style="width: 14px; height: 14px; background: linear-gradient(180deg, #ffc517, #c49000); border-radius: 3px;"></div>
      <span style="color: var(--vp-c-text-2);">HBM Write</span>
    </div>
    <div style="display: flex; align-items: center; gap: 6px;">
      <div style="width: 14px; height: 14px; background: linear-gradient(180deg, #76B900, #5a8a00); border-radius: 3px;"></div>
      <span style="color: var(--vp-c-text-2);">Register Traffic</span>
    </div>
  </div>
  <p style="text-align: center; font-size: 12px; color: var(--vp-c-text-3); margin-top: 8px;">Memory traffic comparison per forward pass (`fused_rmsnorm_rope`, batch=2, seq_len=2048)</p>
</div>

---

<p style="font-size: 12px; color: var(--vp-c-text-3);">
<strong>Data source:</strong> Measured on NVIDIA A100 SXM4 80GB, CUDA 12.1, PyTorch 2.1, Triton 2.1. Latency measured with <code>torch.cuda.synchronize()</code> before and after the timed region, 10 warmup runs + 100 benchmark runs.
</p>
