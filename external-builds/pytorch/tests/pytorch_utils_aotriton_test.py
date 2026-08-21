#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for enable_aotriton_experimental_archs in pytorch_utils.py"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from pytorch_utils import (
    AOTRITON_EXPERIMENTAL_ARCHS,
    enable_aotriton_experimental_archs,
)

ENV_VAR = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


class TestEnableAotritonExperimentalArchs(unittest.TestCase):
    def _run(self, archs, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=True):
            enable_aotriton_experimental_archs(archs)
            return os.environ.get(ENV_VAR)

    def test_experimental_arch_enables_opt_in(self):
        self.assertEqual(self._run(["gfx1151"]), "1")

    def test_all_known_experimental_archs_enable_opt_in(self):
        for arch in AOTRITON_EXPERIMENTAL_ARCHS:
            with self.subTest(arch=arch):
                self.assertEqual(self._run([arch]), "1")

    def test_feature_suffixes_are_stripped(self):
        self.assertEqual(self._run(["gfx950:sramecc+:xnack-"]), "1")

    def test_fully_supported_arch_is_untouched(self):
        for arch in ["gfx90a", "gfx942", "gfx1100"]:
            with self.subTest(arch=arch):
                self.assertIsNone(self._run([arch]))

    def test_no_archs_is_untouched(self):
        self.assertIsNone(self._run([]))

    def test_mixed_archs_enable_opt_in(self):
        self.assertEqual(self._run(["gfx942", "gfx1201"]), "1")

    def test_explicit_user_opt_out_is_preserved(self):
        self.assertEqual(self._run(["gfx1151"], env={ENV_VAR: "0"}), "0")

    def test_explicit_user_opt_in_is_preserved(self):
        self.assertEqual(self._run(["gfx1151"], env={ENV_VAR: "1"}), "1")


if __name__ == "__main__":
    unittest.main()
