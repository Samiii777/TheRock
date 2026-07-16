# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""gfx1151-aware scaled_dot_product_attention (SDPA) layout guard.

Background
----------
The common attention pattern in diffusion models (see HuggingFace Diffusers
``attention_dispatch._native_attention``) takes Q/K/V laid out as
``[batch, seq, heads, head_dim]`` and calls SDPA on *permuted views*::

    q2, k2, v2 = (x.permute(0, 2, 1, 3) for x in (q, k, v))
    out = F.scaled_dot_product_attention(q2, k2, v2)
    out = out.permute(0, 2, 1, 3)

Those ``permute`` calls do not copy — SDPA receives *non-contiguous*
``[batch, heads, seq, head_dim]`` tensors and reads them with strided access.

On gfx1151 (Strix Halo / Radeon 8060S, RDNA3.5) the AOTriton flash-attention
kernel handles the strided reads fine at small sequence lengths, but on the
unified-memory subsystem the strided access causes bandwidth/cache pressure that
scales super-linearly with the sequence length. Past a threshold it becomes
cheaper to pay a one-time contiguous copy than to let the kernel chase strides.

Measured on gfx1151 (torch 2.12.0+rocm7.14, B=1 H=24 D=128, flash SDPA via
``TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1``), contiguous-vs-permute speedup:

    seqlen   speedup (contiguous - permute)
      2048        -38.1 %   (permute wins: copy overhead dominates)
      4096        -21.4 %
      6144        -13.0 %
      6578         +2.8 %   <-- crossover
      7168         +5.1 %
      8192        +11.8 %
      9216        +16.1 %

The issue reporter measured the crossover nearer ~5000 on a newer
torch 2.14/rocm7.15 build (up to +41 % at seqlen ~9800). On a dGPU
(gfx1201, dedicated VRAM, higher bandwidth) the kernel handles the strided
inputs efficiently and the copy is *never* worth it, so the guard must be gated
on gfx1151 only.

This module provides a drop-in replacement for the permute pattern that
materializes contiguous Q/K/V only when it is expected to be a net win.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.nn.functional as F

__all__ = [
    "SDPA_CONTIGUOUS_SEQLEN_THRESHOLD",
    "is_gfx1151",
    "should_materialize_contiguous",
    "sdpa_bshd",
]

# Default sequence-length threshold at/above which materializing contiguous
# Q/K/V is a net win on gfx1151. Chosen conservatively at 6144 so that the
# small-sequence copy penalty is never paid on either the rocm7.14 build
# (crossover ~6500) or the rocm7.15 build (crossover ~5000). Override with the
# THEROCK_SDPA_CONTIGUOUS_SEQLEN_THRESHOLD environment variable.
SDPA_CONTIGUOUS_SEQLEN_THRESHOLD = int(
    os.environ.get("THEROCK_SDPA_CONTIGUOUS_SEQLEN_THRESHOLD", "6144")
)


@functools.lru_cache(maxsize=None)
def is_gfx1151(device_index: int = 0) -> bool:
    """Return True if the given CUDA/HIP device is a gfx1151 part.

    The check is intentionally a substring match on ``gcnArchName`` so that
    variants such as ``gfx1151`` reported with feature suffixes still match,
    while gfx1200/gfx1201 dGPUs (which regress with the copy) do not.
    """
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(device_index).gcnArchName
    except (AssertionError, RuntimeError):
        return False
    return "gfx1151" in arch


def should_materialize_contiguous(
    seqlen: int,
    device_index: int = 0,
    threshold: int | None = None,
) -> bool:
    """Decide whether to pre-materialize contiguous Q/K/V for SDPA.

    Returns True only on gfx1151 and only when ``seqlen`` is at or above the
    crossover threshold, where a contiguous copy is cheaper than strided
    kernel access. On every other device (including gfx1201 dGPUs) this returns
    False, preserving the fast zero-copy permute path.
    """
    if threshold is None:
        threshold = SDPA_CONTIGUOUS_SEQLEN_THRESHOLD
    return is_gfx1151(device_index) and seqlen >= threshold


def sdpa_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    threshold: int | None = None,
    **sdpa_kwargs,
) -> torch.Tensor:
    """SDPA for ``[batch, seq, heads, head_dim]`` (BSHD) inputs with a
    gfx1151-aware contiguous guard.

    This is a drop-in replacement for the diffusion "permute" pattern:
    it accepts Q/K/V in BSHD layout, transposes them to the BHSD layout SDPA
    expects, and returns the result back in BSHD layout.

    On gfx1151 at large sequence lengths the transposed views are materialized
    contiguous (a net win); everywhere else — and at small sequence lengths on
    gfx1151 — the cheaper zero-copy transpose views are used, matching the
    original behavior exactly.
    """
    seqlen = query.shape[1]
    materialize = should_materialize_contiguous(
        seqlen, query.device.index or 0, threshold
    )

    if materialize:
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        return out.transpose(1, 2).contiguous()

    # Fast path: zero-copy permuted views (original behavior).
    q, k, v = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
    return out.permute(0, 2, 1, 3)
