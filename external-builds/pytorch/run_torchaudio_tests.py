#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Runs the torchaudio pytest tree with a bounded per-process memory footprint.

The full suite otherwise runs in one long-lived interpreter whose RSS grows
across tests and, on memory-constrained targets (gfx1150 APUs, docker/cgroup
limit, little swap), gets OOM-killed (exit 137) before report.json is written.

By default each test runs in its own subprocess (pytest-forked), which bounds
peak RSS to a single test regardless of suite size. Tests are also split into
shards so per-shard JSON reports are merged into report.json incrementally; a
shard that produces no report (e.g. OOM-killed) is recorded as incomplete.

    python run_torchaudio_tests.py --test-dir /path/to/audio/test
    python run_torchaudio_tests.py --shards 32 --no-per-test

Env: TORCHAUDIO_TEST_DIR, TORCHAUDIO_TEST_SHARDS, TORCHAUDIO_TEST_MAX_THREADS,
TORCHAUDIO_REPORT_JSON.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_SCRIPT_DIR = Path(__file__).resolve().parent

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# pytest exit codes ranked least -> most severe; used to aggregate over shards.
_EXIT_SEVERITY = {0: 0, 5: 1, 1: 2, 2: 3, 4: 4, 3: 5}


def worse_exit(a: int, b: int) -> int:
    return b if _EXIT_SEVERITY.get(b, 99) > _EXIT_SEVERITY.get(a, 99) else a


def cap_thread_env(env: dict, max_threads: int) -> None:
    for name in _THREAD_ENV_VARS:
        env.setdefault(name, str(max_threads))


def collect_test_ids(test_dir: Path, k_filter: str) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", ".", "--collect-only", "-q",
           "-p", "no:cacheprovider"]
    if k_filter:
        cmd += ["-k", k_filter]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=test_dir)
    ids = [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.strip().startswith(("=", "-"))
    ]
    if not ids and result.returncode not in (0, 5):
        sys.stderr.write(result.stdout + result.stderr)
    return ids


def chunk(items: list, n_shards: int) -> list[list]:
    if not items:
        return []
    n_shards = max(1, min(n_shards, len(items)))
    size = (len(items) + n_shards - 1) // n_shards
    return [items[i : i + size] for i in range(0, len(items), size)]


def _load_report(dst: Path) -> dict:
    merged = json.loads(dst.read_text()) if dst.exists() else {}
    merged.setdefault("tests", [])
    merged.setdefault("summary", {})
    merged.setdefault("incomplete_shards", [])
    merged.setdefault("exitcode", 0)
    return merged


def merge_report(dst: Path, shard_report: Path) -> None:
    shard = json.loads(shard_report.read_text())
    merged = _load_report(dst)
    merged["tests"].extend(shard.get("tests", []))
    for key, value in shard.get("summary", {}).items():
        if isinstance(value, (int, float)):
            merged["summary"][key] = merged["summary"].get(key, 0) + value
    merged["exitcode"] = worse_exit(merged["exitcode"], shard.get("exitcode", 0))
    dst.write_text(json.dumps(merged))


def pre_register(dst: Path, index: int, test_ids: list[str]) -> None:
    merged = _load_report(dst)
    merged["incomplete_shards"].append(
        {"shard": index, "returncode": None, "tests": test_ids}
    )
    dst.write_text(json.dumps(merged))


def finalize_pending(dst: Path, index: int, returncode: int, has_report: bool) -> None:
    merged = _load_report(dst)
    merged["incomplete_shards"] = [
        s for s in merged["incomplete_shards"] if s.get("shard") != index
    ]
    merged["exitcode"] = worse_exit(merged["exitcode"], returncode)
    if not has_report:
        merged["incomplete_shards"].append({"shard": index, "returncode": returncode})
    dst.write_text(json.dumps(merged))


def run_shard(index: int, test_ids: list[str], test_dir: Path, env: dict,
              report_json: Path, per_test: bool) -> int:
    shard_report = report_json.with_suffix(".shard.json")
    shard_report.unlink(missing_ok=True)
    pre_register(report_json, index, test_ids)
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
           "--json-report", f"--json-report-file={shard_report}"]
    if per_test:
        cmd.append("--forked")
    cmd += test_ids
    proc = subprocess.run(cmd, env=env, cwd=test_dir)
    has_report = shard_report.exists()
    if has_report:
        merge_report(report_json, shard_report)
        shard_report.unlink()
    finalize_pending(report_json, index, proc.returncode, has_report)
    return proc.returncode


def cmd_arguments(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--test-dir", type=Path,
        default=Path(os.getenv("TORCHAUDIO_TEST_DIR", THIS_SCRIPT_DIR / "audio" / "test")),
    )
    p.add_argument(
        "--shards", type=int, default=int(os.getenv("TORCHAUDIO_TEST_SHARDS", "16")),
    )
    p.add_argument(
        "--no-per-test", dest="per_test", action="store_false",
        help="Disable per-test subprocess isolation (pytest-forked).",
    )
    p.add_argument(
        "--max-threads", type=int,
        default=int(os.getenv("TORCHAUDIO_TEST_MAX_THREADS", "8")),
    )
    p.add_argument(
        "--report-json", type=Path,
        default=Path(os.getenv("TORCHAUDIO_REPORT_JSON", THIS_SCRIPT_DIR / "report.json")),
    )
    p.add_argument("-k", default="")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = cmd_arguments(argv)
    if not args.test_dir.exists():
        print(f"[ERROR] torchaudio test directory not found: {args.test_dir}")
        return 1

    env = dict(os.environ)
    cap_thread_env(env, args.max_threads)
    args.report_json.unlink(missing_ok=True)

    test_ids = collect_test_ids(args.test_dir, args.k)
    if not test_ids:
        print("[ERROR] no tests collected (see stderr for collection errors)")
        return 1

    shards = chunk(test_ids, args.shards)
    print(f"Collected {len(test_ids)} tests, running in {len(shards)} shards")

    worst = 0
    for i, shard in enumerate(shards):
        print(f"--- shard {i + 1}/{len(shards)}: {len(shard)} tests ---", flush=True)
        rc = run_shard(i, shard, args.test_dir, env, args.report_json, args.per_test)
        print(f"--- shard {i + 1} exit code: {rc} ---", flush=True)
        worst = worse_exit(worst, rc)

    print(f"Torchaudio run finished with exit code {worst}; report at {args.report_json}")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))