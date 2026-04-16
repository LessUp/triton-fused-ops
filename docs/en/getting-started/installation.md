---
layout: default
title: "Installation — Triton Fused Ops"
description: "Installation guide for Triton Fused Ops - system requirements and setup instructions"
---

# 📦 Installation Guide

This guide covers the installation of Triton Fused Ops and its dependencies.

---

## ✅ System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| **GPU** | NVIDIA Ampere (SM80) | NVIDIA H100, A100, or RTX 4090 |
| **VRAM** | 8 GB | 16 GB+ for large models |
| **CUDA** | 11.8 | 12.1+ |

### Software Requirements

| Dependency | Version | Notes |
|:-----------|:--------|:------|
| **Python** | ≥ 3.9 | Python 3.10 or 3.11 recommended |
| **PyTorch** | ≥ 2.0.0 | With CUDA support |
| **Triton** | ≥ 2.1.0 | OpenAI Triton |
| **NumPy** | ≥ 1.21.0 | Required for tensor operations |

---

## 🚀 Installation Methods

### Method 1: Development Installation (Recommended)

For the latest features and development:

```bash
# Clone the repository
git clone https://github.com/LessUp/triton-fused-ops.git
cd triton-fused-ops

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Method 2: Core Installation Only

For production use with minimal dependencies:

```bash
pip install -e .
```

### Method 3: Using uv (Fast)

If you use `uv` for package management:

```bash
# With dev dependencies
uv pip install -e ".[dev]"

# Core only
uv pip install -e .
```

---

## 🔧 Verifying Installation

### Check CUDA Availability

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Test Triton Fused Ops

```python
import torch
from triton_ops import fused_rmsnorm_rope

# Test basic functionality
x = torch.randn(2, 128, 4096, device='cuda', dtype=torch.float16)
weight = torch.ones(4096, device='cuda', dtype=torch.float16)
cos = torch.randn(128, 64, device='cuda', dtype=torch.float16)
sin = torch.randn(128, 64, device='cuda', dtype=torch.float16)

output = fused_rmsnorm_rope(x, weight, cos, sin)
print(f"✅ Output shape: {output.shape}")
print(f"✅ Output dtype: {output.dtype}")
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=triton_ops

# Run specific test file
pytest tests/test_fp8_gemm.py -v
```

---

## 🐛 Troubleshooting

### Issue: "CUDA is not available"

**Cause:** PyTorch installed without CUDA support or CUDA not properly configured.

**Solution:**
```bash
# Reinstall PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Triton not found"

**Cause:** Triton package not installed.

**Solution:**
```bash
pip install triton>=2.1.0
```

### Issue: Import errors with compiled extensions

**Cause:** Version mismatch between PyTorch and Triton.

**Solution:**
```bash
# Upgrade both to latest compatible versions
pip install --upgrade torch triton
```

---

## 📋 GPU Architecture Support

| Architecture | FP16 | FP8 | Notes |
|:-------------|:----:|:---:|:------|
| Ampere (A100) | ✅ | ⚠️ Emulated | Production ready |
| Ada (RTX 4090) | ✅ | ✅ | Edge deployment |
| Hopper (H100) | ✅ | ✅ | Best for FP8 |

---

## 🎯 Next Steps

- [Quick Start Guide](./quickstart.md) — Run your first fused kernel
- [Examples](./examples.md) — Learn from practical examples
- [API Reference](../api/kernels.md) — Explore the API

---

<div align="center">

**[⬆ Back to Top](#-installation-guide)** | **[← Back to Documentation](../)**

</div>
