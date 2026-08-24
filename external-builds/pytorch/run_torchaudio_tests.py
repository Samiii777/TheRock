#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the torchaudio test suite on AMD GPUs with a bounded memory footprint.

Running the whole torchaudio tree in a single pytest process makes peak RSS the
sum of every heavyweight test's high-water mark, because freed heap pages are
not returned to the OS (see torchaudio_memory_plugin).  On unified-memory APUs,
where device allocations are also host allocations, that overruns the container
memory limit and the kernel SIGKILLs the interpreter: the run dies with exit
137, no traceback, and no JSON report for downstream ingestion.

This runner keeps the footprint bounded and the result reportable:

  * each test directory runs in its own pytest process, so a heavy test's peak
    is never stacked on top of an earlier one,
  * torchaudio_memory_plugin trims the heap after every test, so RSS tracks the
    current test rather than the worst one seen so far,
  * a group killed by the OOM killer is recorded as an ``oom-killed`` group and
    the remaining groups still run, and
  * the merged JSON report is always written, even when a group is SIGKILLed.

Usage examples:

    # Run the whole suite:
    python run_torchaudio_tests.py --torchaudio-dir /path/to/audio

    # Run shard 2 of 4:
    python run_torchaudio_tests.py --shard 2 --num-shards 4

    # Run only some groups:
    python run_torchaudio_tests.py --include models functional

    # Pass extra pytest arguments after "--":
    python run_torchaudio_tests.py -- -x --tb=short

Exit codes:
    0 : all tests passed
    1 : test failures, collection errors, or a group killed by the OOM killer
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent

# Per-test function timeout.  Matches run_pytorch_tests_full.py.
PYTEST_TIMEOUT_SECONDS = 900

# Killed-by-SIGKILL shows up as -9 from subprocess and as 137 through a shell.
SIGKILL_RETURNCODES = (-9, 137)

# Test groups, in the order they are executed.  Each entry is a path under
# test/torchaudio_unittest/ and runs in its own process.  The heavyweight model
# tests are split out from the rest of "models" because a single torchscript
# consistency test on the large wav2vec2/hubert configurations peaks at several
# GiB on its own.
TEST_GROUPS: list[str] = [
    "models/wav2vec2",
    "models/tacotron2",
    "models/conformer",
    "models",
    "functional",
    "transforms",
    "compliance",
    "datasets",
    "example",
]


def setup_env() -> None:
    os.environ.setdefault("PYTORCH_TEST_WITH_ROCM", "1")
    os.environ.setdefault("PYTORCH_TESTING_DEVICE_ONLY_FOR", "cuda")
    os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = tempfile.mkdtemp()

    # A high thread count only multiplies the transient per-thread allocations
    # on a memory constrained runner; torchaudio's tests are not thread scaling
    # bound.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "4")

    plugin_dir = str(THIS_SCRIPT_DIR)
    old_pythonpath = os.getenv("PYTHONPATH", "")
    if old_pythonpath:
        os.environ["PYTHONPATH"] = os.pathsep.join([plugin_dir, old_pythonpath])
    else:
        os.environ["PYTHONPATH"] = plugin_dir


def select_groups(test_root: Path, include: list[str]) -> list[str]:
    """Return the groups to run, dropping any that are absent from the tree."""
    selected = []
    for group in include or TEST_GROUPS:
        if (test_root / "torchaudio_unittest" / group).exists():
            selected.append(group)
        else:
            print(f"Skipping group {group!r}: not present in {test_root}")
    return selected


def shard_groups(groups: list[str], shard: int, num_shards: int) -> list[str]:
    if shard <= 0 or num_shards <= 1:
        return groups
    return [g for i, g in enumerate(groups) if i % num_shards == shard - 1]


def build_pytest_cmd(
    group: str,
    report_path: Path,
    cache: bool,
    passthrough_args: list[str],
) -> list[str]:
    """Build the pytest command for one group.

    Groups listed before a parent directory in TEST_GROUPS have already run, so
    the parent ignores them rather than running them a second time.
    """
    target = Path("torchaudio_unittest") / group
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(target),
        "-v",
        "--timeout",
        str(PYTEST_TIMEOUT_SECONDS),
        "-p",
        "torchaudio_memory_plugin",
        "--json-report",
        f"--json-report-file={report_path}",
    ]
    for other in TEST_GROUPS:
        if other != group and other.startswith(group + "/"):
            cmd.extend(["--ignore", str(Path("torchaudio_unittest") / other)])
    if not cache:
        cmd.extend(["-p", "no:cacheprovider"])
    cmd.extend(passthrough_args)
    return cmd


def summarize_group(report_path: Path) -> dict:
    """Read a group's JSON report, tolerating a report that was never written."""
    try:
        with open(report_path) as f:
            report = json.load(f)
    except (OSError, ValueError):
        return {}
    return report.get("summary", {})


def run_group(
    group: str,
    args: argparse.Namespace,
    passthrough_args: list[str],
    report_dir: Path,
) -> dict:
    """Run one group in its own process and return its result record."""
    report_path = report_dir / f"{group.replace('/', '_')}.json"
    cmd = build_pytest_cmd(group, report_path, args.cache, passthrough_args)

    print(f"\n{'=' * 60}")
    print(f"torchaudio group: {group}")
    print(f"{'=' * 60}")
    print(f"Executing: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, cwd=str(args.torchaudio_dir / "test"))
    summary = summarize_group(report_path)

    record = {
        "group": group,
        "returncode": result.returncode,
        "summary": summary,
        "report": str(report_path) if summary else None,
    }

    if result.returncode in SIGKILL_RETURNCODES:
        # The kernel OOM killer SIGKILLs the interpreter, so pytest never runs
        # its session-finish hook and the group's report is missing or partial.
        # Record it explicitly instead of letting the run look like a hang.
        record["status"] = "oom-killed"
        print(
            f"Group {group!r} was killed (return code {result.returncode}); "
            "this is characteristic of the OOM killer. Continuing with the "
            "remaining groups.",
            flush=True,
        )
    elif result.returncode == 0:
        record["status"] = "passed"
    else:
        record["status"] = "failed"

    print(f"Group {group!r} finished: {record['status']} ({summary})", flush=True)
    return record


def write_merged_report(records: list[dict], report_file: Path) -> None:
    """Write the merged report.

    Always produces a file, including when a group was SIGKILLed, so downstream
    ingestion has a structured result instead of a missing report.
    """
    totals: dict[str, int] = {}
    for record in records:
        for key, value in record["summary"].items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value

    merged = {
        "summary": totals,
        "groups": records,
        "oom_killed_groups": [
            r["group"] for r in records if r["status"] == "oom-killed"
        ],
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nWrote merged report to {report_file}")


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run the torchaudio test suite with a bounded memory footprint."
    )
    parser.add_argument(
        "--torchaudio-dir",
        type=Path,
        default=Path(os.getenv("TORCHAUDIO_DIR", THIS_SCRIPT_DIR / "torchaudio")),
        help="Path to the torchaudio source checkout (contains test/).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help=f"Groups to run. Default: {' '.join(TEST_GROUPS)}",
    )
    parser.add_argument(
        "--shard",
        type=int,
        default=int(os.getenv("SHARD_NUMBER", "0")),
        help="1-based shard number.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=int(os.getenv("NUM_TEST_SHARDS", "0")),
        help="Total number of shards.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        default=Path(os.getenv("TORCHAUDIO_REPORT_FILE", "report.json")),
        help="Path of the merged JSON report.",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the pytest cache provider.",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str]) -> int:
    args, passthrough_args = cmd_arguments(argv)
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    test_root = args.torchaudio_dir / "test"
    if not test_root.exists():
        print(f"ERROR: no torchaudio test directory at {test_root}")
        return 1

    setup_env()

    groups = shard_groups(
        select_groups(test_root, args.include), args.shard, args.num_shards
    )
    if not groups:
        print("ERROR: no test groups selected")
        return 1
    print(f"Running {len(groups)} torchaudio group(s): {', '.join(groups)}")

    report_dir = Path(tempfile.mkdtemp(prefix="torchaudio-reports-"))
    records = [run_group(g, args, passthrough_args, report_dir) for g in groups]
    write_merged_report(records, args.report_file)

    print(f"\n{'=' * 60}")
    for record in records:
        print(f"  {record['group']:<24} {record['status']}")
    print(f"{'=' * 60}")

    return 0 if all(r["status"] == "passed" for r in records) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))