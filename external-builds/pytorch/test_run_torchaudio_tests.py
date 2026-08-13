#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for run_torchaudio_tests."""

import json
import textwrap

import run_torchaudio_tests as rt


def test_worse_exit_ranks_by_severity():
    assert rt.worse_exit(0, 5) == 5
    assert rt.worse_exit(5, 1) == 1
    assert rt.worse_exit(1, 137) == 137  # unknown (OOM) beats a plain failure
    assert rt.worse_exit(137, 1) == 137  # order-independent


def test_chunk_bounds():
    shards = rt.chunk(list(range(64)), 16)
    assert len(shards) == 16
    assert sum(len(s) for s in shards) == 64
    assert rt.chunk([1, 2, 3], 16) == [[1], [2], [3]]
    assert rt.chunk([], 16) == []


def test_merge_report_sums_numeric_and_worsens_exit(tmp_path):
    dst = tmp_path / "report.json"
    for i, code in enumerate((0, 1)):
        shard = tmp_path / f"s{i}.json"
        shard.write_text(json.dumps({
            "tests": [{"nodeid": f"t{i}"}],
            "summary": {"passed": 2, "root": "x"},
            "exitcode": code,
        }))
        rt.merge_report(dst, shard)
    merged = json.loads(dst.read_text())
    assert len(merged["tests"]) == 2
    assert merged["summary"]["passed"] == 4
    assert "root" not in merged["summary"]
    assert merged["exitcode"] == 1


def test_pre_register_and_finalize_by_index(tmp_path):
    dst = tmp_path / "report.json"
    rt.pre_register(dst, 0, ["a::t1"])
    rt.pre_register(dst, 1, ["a::t1"])  # identical ids, different shard
    rt.finalize_pending(dst, 0, 0, has_report=True)  # success drops shard 0
    rt.finalize_pending(dst, 1, 137, has_report=False)  # OOM keeps shard 1
    merged = json.loads(dst.read_text())
    shards = {s["shard"]: s for s in merged["incomplete_shards"]}
    assert 0 not in shards
    assert shards[1]["returncode"] == 137
    assert merged["exitcode"] == 137


def test_cap_thread_env_respects_existing():
    env = {"OMP_NUM_THREADS": "2"}
    rt.cap_thread_env(env, 8)
    assert env["OMP_NUM_THREADS"] == "2"
    assert env["MKL_NUM_THREADS"] == "8"


def test_collect_and_run_end_to_end(tmp_path):
    """Exercises collect_test_ids -> run_shard -> report merge together, and
    verifies a failing shard's exit code propagates to the merged report."""
    test_dir = tmp_path / "audio" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "test_sample.py").write_text(textwrap.dedent("""
        def test_pass(): assert True
        def test_fail(): assert False
    """))
    report = tmp_path / "report.json"

    ids = rt.collect_test_ids(test_dir, "")
    assert len(ids) == 2

    import os
    worst = 0
    for i, shard in enumerate(rt.chunk(ids, 2)):
        rc = rt.run_shard(i, shard, test_dir, dict(os.environ), report, per_test=False)
        worst = rt.worse_exit(worst, rc)

    assert worst == 1  # the failing test must surface, not be masked
    merged = json.loads(report.read_text())
    assert merged["summary"].get("passed") == 1
    assert merged["summary"].get("failed") == 1
    assert merged["incomplete_shards"] == []
