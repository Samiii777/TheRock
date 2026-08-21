# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the shared PyTorch test-runner helpers in ``pytorch_utils.py``."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_utils import (
    AOTRITON_EXPERIMENTAL_ARCHS,
    enable_aotriton_experimental_archs,
)

ENV_VAR = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


@pytest.mark.parametrize("arch", sorted(AOTRITON_EXPERIMENTAL_ARCHS))
def test_enables_experimental_arch(arch):
    enable_aotriton_experimental_archs([arch])
    assert os.environ[ENV_VAR] == "1"


def test_strips_arch_feature_suffix():
    enable_aotriton_experimental_archs(["gfx1151:sramecc+:xnack-"])
    assert os.environ[ENV_VAR] == "1"


@pytest.mark.parametrize("archs", [[], ["gfx942"], ["gfx90a", "gfx1100"]])
def test_leaves_non_experimental_archs_untouched(archs):
    enable_aotriton_experimental_archs(archs)
    assert ENV_VAR not in os.environ


def test_enables_on_mixed_arch_selection():
    enable_aotriton_experimental_archs(["gfx942", "gfx1151"])
    assert os.environ[ENV_VAR] == "1"


@pytest.mark.parametrize("value", ["0", "1"])
def test_preserves_explicit_setting(monkeypatch, value):
    monkeypatch.setenv(ENV_VAR, value)
    enable_aotriton_experimental_archs(["gfx1151"])
    assert os.environ[ENV_VAR] == value
