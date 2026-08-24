# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the torchaudio runner's OOM handling and memory reclaim.

These cover the behaviour that keeps an OOM-killed group from destroying the
whole run: a SIGKILLed group must be reported as ``oom-killed``, must not stop
the remaining groups, and the merged report must still be written.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, MODULE_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_torchaudio_tests")
memory_plugin = _load("torchaudio_memory_plugin")


def _record(group, status, summary=None):
    return {
        "group": group,
        "status": status,
        "returncode": 0,
        "summary": summary or {},
        "report": None,
    }


def test_sigkill_returncodes_cover_signal_and_shell_forms():
    assert -9 in runner.SIGKILL_RETURNCODES
    assert 137 in runner.SIGKILL_RETURNCODES


def test_heavy_model_dirs_run_before_their_parent():
    groups = runner.TEST_GROUPS
    for group in groups:
        if group.startswith("models/"):
            assert groups.index(group) < groups.index(
                "models"
            ), f"{group} must run before the parent models group"


def test_parent_group_ignores_subdirectories_already_run(tmp_path):
    cmd = runner.build_pytest_cmd("models", tmp_path / "r.json", cache=False, passthrough_args=[])
    for group in runner.TEST_GROUPS:
        if group.startswith("models/"):
            assert str(Path("torchaudio_unittest") / group) in cmd
    assert cmd.count("--ignore") == sum(
        1 for g in runner.TEST_GROUPS if g.startswith("models/")
    )


def test_build_pytest_cmd_loads_the_memory_plugin(tmp_path):
    cmd = runner.build_pytest_cmd("functional", tmp_path / "r.json", cache=True, passthrough_args=[])
    assert "torchaudio_memory_plugin" in cmd
    assert "no:cacheprovider" not in cmd


def test_build_pytest_cmd_appends_passthrough_args(tmp_path):
    cmd = runner.build_pytest_cmd(
        "functional", tmp_path / "r.json", cache=False, passthrough_args=["-x", "--tb=short"]
    )
    assert cmd[-2:] == ["-x", "--tb=short"]
    assert "no:cacheprovider" in cmd


def test_summarize_group_tolerates_a_report_that_was_never_written(tmp_path):
    assert runner.summarize_group(tmp_path / "missing.json") == {}


def test_summarize_group_tolerates_a_truncated_report(tmp_path):
    partial = tmp_path / "partial.json"
    partial.write_text('{"summary": {"passed": 1')
    assert runner.summarize_group(partial) == {}


def test_merged_report_is_written_when_a_group_was_oom_killed(tmp_path):
    report_file = tmp_path / "nested" / "report.json"
    records = [
        _record("models/wav2vec2", "oom-killed"),
        _record("models/conformer", "passed", {"passed": 4, "total": 4}),
        _record("functional", "failed", {"passed": 8, "failed": 2, "total": 10}),
    ]
    runner.write_merged_report(records, report_file)

    merged = json.loads(report_file.read_text())
    assert merged["oom_killed_groups"] == ["models/wav2vec2"]
    assert merged["summary"] == {"passed": 12, "total": 14, "failed": 2}
    assert [g["group"] for g in merged["groups"]] == [
        "models/wav2vec2",
        "models/conformer",
        "functional",
    ]


def test_shard_groups_partitions_without_loss_or_overlap():
    groups = runner.TEST_GROUPS
    num_shards = 4
    shards = [runner.shard_groups(groups, i, num_shards) for i in range(1, num_shards + 1)]
    flattened = [g for shard in shards for g in shard]
    assert sorted(flattened) == sorted(groups)
    assert len(flattened) == len(set(flattened))


@pytest.mark.parametrize("shard,num_shards", [(0, 0), (0, 4), (1, 1)])
def test_shard_groups_returns_everything_when_sharding_is_disabled(shard, num_shards):
    assert runner.shard_groups(runner.TEST_GROUPS, shard, num_shards) == runner.TEST_GROUPS


def test_select_groups_drops_groups_missing_from_the_tree(tmp_path):
    (tmp_path / "torchaudio_unittest" / "functional").mkdir(parents=True)
    assert runner.select_groups(tmp_path, ["functional", "not_there"]) == ["functional"]


def test_setup_env_caps_thread_counts_without_overriding_the_caller(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    runner.setup_env()
    assert os.environ["OMP_NUM_THREADS"] == "64"
    assert os.environ["MKL_NUM_THREADS"] == "4"


def test_setup_env_puts_the_plugin_directory_on_pythonpath(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    runner.setup_env()
    entries = os.environ["PYTHONPATH"].split(os.pathsep)
    assert entries[0] == str(runner.THIS_SCRIPT_DIR)
    assert "/somewhere/else" in entries


def test_release_freed_memory_returns_heap_pages_to_the_os():
    baseline = memory_plugin.current_rss_mb()
    blocks = [bytearray(8 * 1024 * 1024) for _ in range(64)]
    peak = memory_plugin.current_rss_mb()
    assert peak > baseline + 256

    del blocks
    memory_plugin.release_freed_memory()
    assert memory_plugin.current_rss_mb() < peak - 256


def test_release_freed_memory_is_safe_without_malloc_trim(monkeypatch):
    monkeypatch.setattr(memory_plugin, "_MALLOC_TRIM", None)
    memory_plugin.release_freed_memory()