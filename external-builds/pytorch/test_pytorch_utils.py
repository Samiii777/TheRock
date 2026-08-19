# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the AOTriton experimental-arch opt-in in ``pytorch_utils``."""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pytorch_utils import (
    AOTRITON_EXPERIMENTAL_ARCHS,
    enable_aotriton_experimental_archs,
)

VAR = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(VAR, raising=False)


def test_experimental_arch_sets_opt_in():
    assert enable_aotriton_experimental_archs(["gfx1151"]) is True
    assert os.environ[VAR] == "1"


def test_supported_arch_is_untouched():
    assert enable_aotriton_experimental_archs(["gfx942", "gfx90a", "gfx1100"]) is False
    assert VAR not in os.environ


def test_no_visible_archs():
    assert enable_aotriton_experimental_archs([]) is False
    assert VAR not in os.environ


def test_feature_suffixes_are_stripped():
    assert enable_aotriton_experimental_archs(["gfx950:sramecc+:xnack-"]) is True
    assert os.environ[VAR] == "1"


def test_mixed_archs_opt_in():
    assert enable_aotriton_experimental_archs(["gfx942", "gfx1201"]) is True
    assert os.environ[VAR] == "1"


def test_explicit_opt_out_is_preserved(monkeypatch):
    monkeypatch.setenv(VAR, "0")
    assert enable_aotriton_experimental_archs(["gfx1151"]) is True
    assert os.environ[VAR] == "0"


def test_fully_supported_archs_excluded():
    assert AOTRITON_EXPERIMENTAL_ARCHS.isdisjoint({"gfx90a", "gfx942", "gfx1100"})
