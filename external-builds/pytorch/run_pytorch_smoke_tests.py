# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PyTorch ROCm Smoke Tests Runner.

This script runs PyTorch smoke tests using pytest.

Usage Examples
--------------
Basic usage (auto-detect GPU):
    $ python run_pytorch_smoke_tests.py

Specify GPU family:
    $ python run_pytorch_smoke_tests.py --amdgpu-family gfx942

Pass additional pytest arguments after "--":
    $ python run_pytorch_smoke_tests.py -- -m "slow"
    $ python run_pytorch_tests.py -- --tb=short -x
"""

import argparse
import os
import sys

from pathlib import Path

import pytest

from pytorch_utils import (
    get_unique_supported_devices,
)

THIS_SCRIPT_DIR = Path(__file__).resolve().parent


def cmd_arguments(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse command line arguments.

    Args:
        argv: Command line arguments (without program name).

    Returns:
        Tuple of (parsed args, passthrough pytest args passed after "--").
    """
    # Extract passthrough pytest args after "--"
    try:
        rest_pos = argv.index("--")
    except ValueError:
        passthrough_pytest_args = []
    else:
        passthrough_pytest_args = argv[rest_pos + 1 :]
        argv = argv[:rest_pos]

    parser = argparse.ArgumentParser(
        description="""
Runs PyTorch smoke-tests for AMD GPUs.
All arguments after "--" are passed directly to pytest (e.g. -p no:cacheprovider).
"""
    )

    parser.add_argument(
        "--amdgpu-family",
        type=str,
        default=os.getenv("AMDGPU_FAMILY", ""),
        required=False,
        help="""Amdgpu family (e.g. "gfx942").
Select (potentially) additional tests to be skipped based on the amdgpu family""",
    )

    args = parser.parse_args(argv)

    return args, passthrough_pytest_args


def main() -> int:
    """Main entry point for the PyTorch smoke-tests runner.

    Returns:
        Exit code from pytest (0 for success, non-zero for failures).
        Returns non-zero if any device's tests fail.
    """
    try:
        args, passthrough_pytest_args = cmd_arguments(sys.argv[1:])

        # Assumes that the smoke-tests are located in the same directory as this script
        smoke_tests_dir = THIS_SCRIPT_DIR / "smoke-tests"

        if not smoke_tests_dir.exists():
            print(f"ERROR: Directory at '{smoke_tests_dir}' does not exist.")
            sys.exit(1)

        # Detect the unique supported devices ONCE up front and then iterate
        # over them. The device-detection helpers spawn a subprocess that
        # imports torch and respects the HIP_VISIBLE_DEVICES env var. If we
        # called the detection helper inside the per-device loop, the
        # HIP_VISIBLE_DEVICES we set on iteration N would shrink the device
        # set seen on iteration N+1, causing offset-based selection to fail
        # with "Offset N out of range for ... unique devices" (issue #4889).
        unique_devices = get_unique_supported_devices(args.amdgpu_family, log=True)

        if not unique_devices:
            print("[ERROR] No supported devices found")
            sys.exit(1)

        unique_device_count = len(unique_devices)
        print(f"Will run smoke tests on {unique_device_count} unique device(s)")

        # Track overall success
        overall_retcode = 0

        pytest_args = [
            f"{smoke_tests_dir}",
        ]

        # Append any passthrough pytest args passed after "--"
        pytest_args.extend(passthrough_pytest_args)

        # Run smoke tests for each unique device iteratively. We set
        # HIP_VISIBLE_DEVICES directly per iteration (one device at a time)
        # instead of re-running detection per iteration, which would otherwise
        # be corrupted by the env-var we just set (see issue #4889).
        for arch, device_idx in unique_devices.items():
            os.environ["HIP_VISIBLE_DEVICES"] = str(device_idx)

            print(f"\n{'='*60}")
            print(f"Running smoke tests on device {device_idx} ({arch})")
            print(f"HIP_VISIBLE_DEVICES={os.environ['HIP_VISIBLE_DEVICES']}")
            print(f"{'='*60}")

            retcode = pytest.main(pytest_args)
            print(
                f"Pytest finished for device {device_idx} ({arch}) with return code: {retcode}"
            )

            # Track if any test run failed
            if retcode != 0:
                overall_retcode = retcode

        print(f"\n{'='*60}")
        print(f"All smoke tests completed. Overall return code: {overall_retcode}")
        print(f"{'='*60}")

        return overall_retcode

    except (ValueError, IndexError) as e:
        print(f"[ERROR] Exception in PyTorch smoke-tests runner: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Lets make this script return pytest exit code (success or failure)
    sys.exit(main())
