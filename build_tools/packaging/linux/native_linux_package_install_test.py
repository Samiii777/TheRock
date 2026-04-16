#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Post-install verification for ROCm native packages (no installation).

Run ``native_linux_package_install.py`` first to configure repos and install packages
(or install ROCm by other means). This script only checks the install prefix,
installed package listing, rocminfo, and optionally ``rdhc.py --all``.

``native_linux_package_install.py`` also provides ``--simulate-packages-dir`` for
dry-run local *.deb/*.rpm; that is not verification and lives with install.

Sample use cases for **this script** (``native_linux_package_install_test.py``):

- **Sanity (default):** verify prefix, key files, dpkg/rpm listing, rocminfo.
  ``--test-type sanity``
- **Full:** same as sanity plus RDHC. ``--test-type full``

Required: ``--os-profile`` (selects dpkg vs rpm vs zypper for queries) and
``--install-prefix`` (ROCm tree to verify).

Path overrides: ROCM_RDHC_REL_PATH (relative path from install prefix to rdhc).

Prerequisites:
- Run inside the same container/VM where ROCm is installed.
- Python packages for RDHC: see build_tools/packaging/linux/tests/requirements.txt

Example invocations:

 # Basic verification after a manual or scripted install
 python3 native_linux_package_install_test.py \\
   --os-profile ubuntu2404 \\
   --install-prefix /opt/rocm/core

 # Basic + RDHC
 python3 native_linux_package_install_test.py --test-type full \\
   --os-profile rhel8 \\
   --install-prefix /opt/rocm/core

 # Typical two-step CI: install then verify (separate steps)
 python3 native_linux_package_install.py --os-profile ubuntu2404 \\
   --repo-url https://.../ --gfx-arch gfx94x --release-type nightly
 python3 native_linux_package_install_test.py \\
   --os-profile ubuntu2404 --install-prefix /opt/rocm/core
"""

import argparse
import importlib.util
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path


def _load_native_linux_package_install():
    """Load sibling native_linux_package_install.py without mutating sys.path."""
    install_path = Path(__file__).resolve().parent / "native_linux_package_install.py"
    if not install_path.is_file():
        raise FileNotFoundError(
            f"Expected native_linux_package_install.py beside this script: {install_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "native_linux_package_install",
        install_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {install_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_install_mod = _load_native_linux_package_install()
NativeLinuxPackageInstaller = _install_mod.NativeLinuxPackageInstaller
_env = _install_mod._env

VERIFY_KEY_COMPONENTS = [
    "bin/rocminfo",
    "bin/hipcc",
    "bin/clinfo",
    "include/hip/hip_runtime.h",
    "lib/libamdhip64.so",
]
RDHC_REL_PATH = _env("ROCM_RDHC_REL_PATH", "libexec/rocm-core/rdhc.py")

ROCMINFO_TIMEOUT_SEC = 30
RDHC_TIMEOUT_SEC = 30
VERIFY_MIN_COMPONENTS = 2


class NativeLinuxPackageVerifier:
    """Verify an existing ROCm install (no repo setup, no package install)."""

    def __init__(self, os_profile: str, install_prefix: str) -> None:
        self.os_profile = os_profile.lower()
        self.install_prefix = install_prefix
        self.package_type = NativeLinuxPackageInstaller._derive_package_type(os_profile)

    def _is_sles(self) -> bool:
        return self.os_profile.startswith("sles")

    def run_basic_verification(self) -> bool:
        """Verify install prefix, key components, package list, rocminfo."""
        print("\n" + "=" * 80)
        print("STEP 1: BASIC INSTALL VERIFICATION")
        print("=" * 80)

        install_path = Path(self.install_prefix)
        if not install_path.exists():
            print(f"\n[FAIL] Installation directory not found: {self.install_prefix}")
            return False

        print(f"\n[PASS] Installation directory exists: {self.install_prefix}")

        key_components = VERIFY_KEY_COMPONENTS
        print("\nChecking for key ROCm components:")
        found_count = 0
        for component in key_components:
            component_path = install_path / component
            if component_path.exists():
                print(f" [PASS] {component}")
                found_count += 1
            else:
                print(f" [WARN] {component} (not found)")

        print(f"\nComponents found: {found_count}/{len(key_components)}")

        print("\nChecking installed packages:")
        try:
            if self.package_type == "deb":
                cmd = ["dpkg", "-l"]
                grep_pattern = "rocm"
            elif self._is_sles():
                cmd = ["zypper", "--non-interactive", "search", "-i", "rocm"]
                grep_pattern = "rocm"
            else:
                cmd = ["rpm", "-qa"]
                grep_pattern = "rocm"

            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            rocm_packages = [
                line
                for line in result.stdout.split("\n")
                if grep_pattern.lower() in line.lower()
            ]
            print(f" Found {len(rocm_packages)} ROCm packages installed")
            if rocm_packages:
                print("\n Sample packages (Show first 5):")
                for pkg in rocm_packages[:5]:
                    print(f" {pkg.strip()}")
                if len(rocm_packages) > 5:
                    print(f" ... and {len(rocm_packages) - 5} more")
        except subprocess.CalledProcessError:
            print(" [WARN] Could not query installed packages")

        rocminfo_path = install_path / "bin" / "rocminfo"
        if rocminfo_path.exists():
            print("\nTrying to run rocminfo...")
            try:
                result = subprocess.run(
                    [str(rocminfo_path)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=ROCMINFO_TIMEOUT_SEC,
                )
                print(" [PASS] rocminfo executed successfully")
                lines = result.stdout.split("\n")[:10]
                print("\n First few lines of rocminfo output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            except subprocess.TimeoutExpired:
                print(" [WARN] rocminfo timed out (may require GPU hardware)")
            except subprocess.CalledProcessError:
                print(" [WARN] rocminfo failed (may require GPU hardware)")
            except OSError as e:
                print(f" [WARN] Could not run rocminfo: {e}")

        if found_count >= VERIFY_MIN_COMPONENTS:
            print("\n[PASS] Basic verification PASSED")
            return True
        print("\n[FAIL] Basic verification FAILED (insufficient components)")
        return False

    def run_full_verification(self) -> bool:
        """Run rdhc.py --all."""
        print("\n" + "=" * 80)
        print("STEP 2: FULL VERIFICATION (RDHC)")
        print("=" * 80)
        return self.test_rdhc()

    def test_rdhc(self) -> bool:
        """Test rdhc.py under install prefix."""
        print("\n" + "=" * 80)
        print("TESTING RDHC.PY")
        print("=" * 80)

        install_path = Path(self.install_prefix).resolve()
        rdhc_script = (install_path / RDHC_REL_PATH).resolve()
        rocm_install_prefix_arg = str(install_path)

        if not rdhc_script.exists():
            print(f"\n[WARN] rdhc.py not found at: {rdhc_script}")
            print(" This is expected if rocm-core package is not installed")
            return False

        print(f"\n[PASS] rdhc.py found at: {rdhc_script}")

        # Always run rdhc with this process's interpreter.
        cmd = [sys.executable, str(rdhc_script)]

        test_args = ["--rocm-install-prefix", rocm_install_prefix_arg, "--all"]
        print(
            f"\nRun rdhc.py with --rocm-install-prefix {rocm_install_prefix_arg} --all..."
        )
        print(f"Command: {' '.join(cmd + test_args)}")

        try:
            result = subprocess.run(
                cmd + test_args,
                cwd=str(install_path),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=RDHC_TIMEOUT_SEC,
            )
            print(" [PASS] rdhc.py executed successfully")
            if result.stdout:
                lines = result.stdout.split("\n")[:5]
                print("\n First few lines of output:")
                for line in lines:
                    if line.strip():
                        print(f" {line}")
            return True
        except subprocess.TimeoutExpired:
            print(" [WARN] rdhc.py --all timed out")
            return False
        except subprocess.CalledProcessError:
            print(" [WARN] rdhc.py --all failed")
            return False
        except OSError as e:
            print(f" [WARN] Could not run rdhc.py: {e}")
            return False


_CLI_EXAMPLES_EPILOG = """
Examples:
 # Verify existing install (sanity)
 python native_linux_package_install_test.py --os-profile ubuntu2404 \\
   --install-prefix /opt/rocm/core

 # Verify + RDHC
 python native_linux_package_install_test.py --test-type full --os-profile rhel8 \\
   --install-prefix /opt/rocm/core
"""


def _build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Verify an existing ROCm native package install (no installation). "
            "Use native_linux_package_install.py to install from a repository."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLI_EXAMPLES_EPILOG,
    )
    parser.add_argument(
        "--os-profile",
        type=str,
        help="OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3). Required for sanity/full; for simulate, used only to derive pkg-type if --pkg-type is omitted.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        help="Full repository URL (constructed in YAML workflow). Required for sanity/full; not used for simulate.",
    )
    parser.add_argument(
        "--gfx-arch",
        type=str,
        nargs="+",
        metavar="ARCH",
        help="GPU architecture(s) as a list. Only the first is used for now. Required for sanity/full; not used for simulate. Examples: gfx94x, gfx110x gfx1151",
    )
    parser.add_argument(
        "--release-type",
        type=str,
        choices=["dev", "nightly", "prerelease", "release", "ci"],
        help="Type of release: 'dev', 'nightly', 'prerelease', 'release', or 'ci'",
    )
    parser.add_argument(
        "--install-prefix",
        type=str,
        required=True,
        help="ROCm installation prefix to verify (e.g. /opt/rocm/core).",
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["sanity", "full"],
        default="sanity",
        help="'sanity' = basic checks only; 'full' = basic + rdhc.py --all.",
    )
    return parser


def parse_cli_arguments(argv: list[str] | None = None) -> Namespace:
    """Build parser, parse argv."""
    parser = _build_argument_parser()
    return parser.parse_args(argv)


def run_tests(args: Namespace) -> int:
    """Run verification from parsed CLI args. Returns exit code (0 success)."""
    try:
        NativeLinuxPackageInstaller._derive_package_type(args.os_profile)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 80)
    print("VERIFICATION CONFIGURATION")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Install Prefix: {args.install_prefix}")
    print(f"Test Type: {args.test_type}")
    print("=" * 80)

    verifier = NativeLinuxPackageVerifier(
        os_profile=args.os_profile,
        install_prefix=args.install_prefix,
    )

    print("\n" + "=" * 80)
    print("NATIVE LINUX PACKAGE VERIFICATION")
    print("=" * 80)

    try:
        if not verifier.run_basic_verification():
            print("\n[FAIL] Basic verification failed.")
            return 1
        if args.test_type == "full":
            if not verifier.run_full_verification():
                print("\n[FAIL] Full verification (RDHC) failed.")
                return 1
        print("\n" + "=" * 80)
        print("[PASS] VERIFICATION PASSED")
        if args.test_type == "sanity":
            print("(sanity: basic checks completed)")
        else:
            print("(full: basic checks and RDHC completed)")
        print("=" * 80 + "\n")
        return 0
    except Exception as e:
        print(f"\n[FAIL] Error during verification: {e}")
        traceback.print_exc()
        return 1


def main() -> None:
    """Entry point: parse CLI, then run verification."""
    args = parse_cli_arguments()
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
