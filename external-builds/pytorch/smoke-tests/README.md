# PyTorch Smoke Tests

This repository contains a set of basic smoke tests for verifying fundamental PyTorch tensor operations. These tests ensure that basic matrix operations function correctly, but they are **not exhaustive PyTorch tests**.

## Disclaimer

⚠️ **These tests are not full PyTorch tests.** They are designed as quick checks for basic tensor operations and do not replace comprehensive testing of PyTorch functionalities.

## Test Overview

The following operations are covered in these smoke tests:

- **Matrix Multiplication (`torch.mm`)**
- **Batch Matrix Multiplication (`torch.bmm`)**
- **Matrix Multiplication using `@` operator**
- **Element-wise Multiplication (`*`)**
- **Matrix Transposition (`torch.t`)**
- **Dot Product (`torch.dot`)**
- **Matrix-Vector Multiplication (`torch.mv`)**
- **General Matrix Multiplication (`torch.matmul`)**
- **Convolution (`torch.conv2d` and `torch.nn.functional.conv_transpose2d`)**

## gfx1151 SDPA contiguous layout guard

`sdpa_contiguous_guard.py` provides `sdpa_bshd`, a drop-in replacement for the
common diffusion attention "permute" pattern
(`x.permute(0, 2, 1, 3)` views passed to
`scaled_dot_product_attention`). On gfx1151 (Strix Halo / Radeon 8060S) the
AOTriton flash-attention kernel handles the resulting non-contiguous inputs
efficiently at small sequence lengths, but strided access causes bandwidth/cache
pressure on the unified-memory subsystem that grows with the sequence length.
Past a threshold (default 6144, override with
`THEROCK_SDPA_CONTIGUOUS_SEQLEN_THRESHOLD`) it becomes cheaper to materialize
contiguous Q/K/V. The guard is gated on gfx1151 only — dGPUs such as gfx1201
regress with the copy and always use the zero-copy path. See TheRock issue
#6584.

Run the regression tests (flash SDPA on gfx11xx requires the experimental env
var):

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
  python -m pytest sdpa_contiguous_guard_test.py -v
```