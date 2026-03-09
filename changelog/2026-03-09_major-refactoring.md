# Major Refactoring - v0.2.0

Date: 2026-03-09

## Critical Bug Fixes

### Gated MLP: gate/up projection activation swapped (Correctness Bug)
- The standard SwiGLU formula is `output = activation(gate_proj(x)) * up_proj(x)`
- The kernel and reference implementation both applied activation to `up_acc` instead of `gate_acc`
- This produced incorrect outputs for any model using SwiGLU (LLaMA, Mistral, etc.)
- Fixed in both the Triton kernel (`fused_gated_mlp_kernel`) and the PyTorch reference (`gated_mlp_reference`)

### RMSNorm kernel: incorrect batch_idx computation
- `rmsnorm_kernel` computed `batch_idx = row_idx // cdiv(hidden_dim, BLOCK_SIZE)` which is mathematically wrong — `hidden_dim / BLOCK_SIZE` has nothing to do with batch indexing
- The variable was unused but indicated a misunderstanding of the program grid
- Removed the bogus computation; `row_idx` from `program_id(0)` already correctly indexes the flattened batch*seq grid

## Performance Improvements

### FP8Linear: weight transpose cached instead of recomputed per-forward
- `FP8Linear.forward()` called `self.weight_fp8.t().contiguous()` on every forward pass — an expensive allocation + copy for large weight matrices
- Now pre-computes and caches the transposed weight as `weight_fp8_t` during `quantize_weights()`
- Eliminates one GPU allocation per forward pass

## Code Quality

### api.py: consolidated duplicate imports
- Each kernel module was imported twice with separate `from...import` blocks
- Consolidated into single import statements per module

## Version
- 0.1.0 → 0.2.0 (pyproject.toml + __init__.py)

### Files Modified
- `triton_ops/kernels/gated_mlp.py` — activation applied to gate, not up (kernel + reference)
- `triton_ops/kernels/fp8_gemm.py` — cached transposed weight in FP8Linear
- `triton_ops/kernels/rmsnorm_rope.py` — removed incorrect batch_idx in rmsnorm_kernel
- `triton_ops/api.py` — consolidated duplicate imports
- `triton_ops/__init__.py` — version bump
- `pyproject.toml` — version bump
