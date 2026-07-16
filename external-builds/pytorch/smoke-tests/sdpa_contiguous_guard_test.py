# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for the gfx1151 SDPA contiguous layout guard.

See ``sdpa_contiguous_guard.py`` and TheRock issue #6584.

These tests verify:
  * the guard's decision logic (gfx1151 + large seqlen => materialize; otherwise
    keep the zero-copy permute path), for both a large non-contiguous case and a
    smaller case where the copy would be a loss; and
  * that the guarded ``sdpa_bshd`` helper produces numerically-correct output on
    both paths (matching a MATH-backend reference).
"""

import pytest
import torch
import torch.nn.functional as F

from sdpa_contiguous_guard import (
    SDPA_CONTIGUOUS_SEQLEN_THRESHOLD,
    is_gfx1151,
    sdpa_bshd,
    should_materialize_contiguous,
)

# ---------------------------------------------------------------------------
# Decision-logic tests (no GPU required — exercised for whatever device is
# present, and force-checked for both the large and small seqlen regimes).
# ---------------------------------------------------------------------------


class TestGuardDecision:
    def test_small_seqlen_keeps_permute_path(self):
        # A small case where the contiguous copy is still a loss: the guard must
        # NOT materialize contiguous tensors regardless of architecture.
        small = SDPA_CONTIGUOUS_SEQLEN_THRESHOLD - 1
        assert (
            should_materialize_contiguous(
                small, threshold=SDPA_CONTIGUOUS_SEQLEN_THRESHOLD
            )
            is False
        )

    def test_large_seqlen_gated_on_gfx1151(self):
        # A large non-contiguous case: the guard materializes contiguous Q/K/V
        # only on gfx1151. On any other device the zero-copy path is kept.
        large = SDPA_CONTIGUOUS_SEQLEN_THRESHOLD + 4096
        decision = should_materialize_contiguous(
            large, threshold=SDPA_CONTIGUOUS_SEQLEN_THRESHOLD
        )
        assert decision == is_gfx1151()

    def test_custom_threshold_override(self):
        # Below custom threshold => never materialize.
        assert should_materialize_contiguous(1000, threshold=2000) is False
        # At/above custom threshold => only on gfx1151.
        assert should_materialize_contiguous(2000, threshold=2000) == is_gfx1151()

    def test_non_gfx1151_never_materializes(self):
        if is_gfx1151():
            pytest.skip("Running on gfx1151; this asserts the dGPU/other-arch path")
        huge = 100_000
        assert should_materialize_contiguous(huge, threshold=1) is False


# ---------------------------------------------------------------------------
# Numerical-correctness tests for the helper (require a GPU).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm/CUDA GPU")
class TestSdpaBshdCorrectness:
    def teardown_method(self):
        torch.cuda.synchronize()

    def _reference_bshd(self, q, k, v):
        # MATH-backend reference computed on the same BSHD->BHSD layout.
        from torch.nn.attention import sdpa_kernel, SDPBackend

        qp, kp, vp = (x.permute(0, 2, 1, 3) for x in (q, k, v))
        with sdpa_kernel(SDPBackend.MATH):
            out = F.scaled_dot_product_attention(qp, kp, vp)
        return out.permute(0, 2, 1, 3)

    @pytest.mark.parametrize(
        "seqlen",
        [
            SDPA_CONTIGUOUS_SEQLEN_THRESHOLD - 2048,  # small: permute path
            SDPA_CONTIGUOUS_SEQLEN_THRESHOLD + 3000,  # large: guard path on gfx1151
        ],
    )
    def test_sdpa_bshd_matches_reference(self, seqlen):
        torch.manual_seed(0)
        b, h, d = 1, 24, 128
        q = torch.randn(b, seqlen, h, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, seqlen, h, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, seqlen, h, d, device="cuda", dtype=torch.float16)

        got = sdpa_bshd(q, k, v)
        ref = self._reference_bshd(q, k, v)

        assert got.shape == ref.shape == (b, seqlen, h, d)
        # fp16 flash vs math: loose tolerance is expected/appropriate.
        torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    def test_guard_path_and_fast_path_agree(self):
        # Force both code paths at the same shape and confirm they agree
        # numerically, proving the contiguous copy does not change semantics.
        torch.manual_seed(0)
        b, s, h, d = 1, 4096, 24, 128
        q = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
        k = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)
        v = torch.randn(b, s, h, d, device="cuda", dtype=torch.float16)

        # threshold=1 forces the contiguous path (on gfx1151); very large
        # threshold forces the permute path everywhere.
        forced_contig = sdpa_bshd(q, k, v, threshold=1)
        forced_permute = sdpa_bshd(q, k, v, threshold=10**9)

        assert forced_contig.shape == forced_permute.shape
        torch.testing.assert_close(forced_contig, forced_permute, atol=2e-2, rtol=2e-2)
