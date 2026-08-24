# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Pytest plugin that returns freed heap pages to the OS between tests.

Python-level frees do not shrink RSS: glibc only returns pages to the kernel
when the top of the heap happens to be contiguous free space, so the large
transient allocations made by torch/ATen leave the process latched at the
high-water mark of the single heaviest test for the rest of the session.  On
unified-memory APUs, where device allocations also come out of host RAM, that
latching is what drives a long run into the OOM killer.  Calling malloc_trim(3)
after each test makes RSS track the current test instead of the worst one.

Enable with ``-p torchaudio_memory_plugin``; set THEROCK_TEST_RSS_LOG to also
record per-test RSS as CSV.
"""

import ctypes
import ctypes.util
import gc
import os
import sys


def _load_malloc_trim():
    """Return glibc's malloc_trim, or None where it is unavailable."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        trim = libc.malloc_trim
    except (OSError, AttributeError):
        return None
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    return trim


_MALLOC_TRIM = _load_malloc_trim()
_PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024 if hasattr(os, "sysconf") else 4


def current_rss_mb() -> float:
    """Resident set size of this process in MiB, or 0.0 if unavailable."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * _PAGE_KB / 1024
    except (OSError, ValueError, IndexError):
        return 0.0


def release_freed_memory() -> None:
    """Drop reference cycles, then hand free heap pages back to the kernel."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


class _MemoryReclaimPlugin:
    def __init__(self, rss_log):
        self._rss_log = rss_log
        self._count = 0

    def pytest_configure(self, config):
        if self._rss_log:
            with open(self._rss_log, "w") as f:
                f.write("n,rss_mb,nodeid\n")

    def pytest_runtest_logfinish(self, nodeid, location):
        release_freed_memory()
        if self._rss_log:
            self._count += 1
            with open(self._rss_log, "a") as f:
                f.write(f"{self._count},{current_rss_mb():.1f},{nodeid}\n")


def pytest_configure(config):
    config.pluginmanager.register(
        _MemoryReclaimPlugin(os.environ.get("THEROCK_TEST_RSS_LOG", "")),
        "therock_memory_reclaim",
    )
