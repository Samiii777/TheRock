#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Repository setup, simulated dry-run install, and ROCm native package installation.

Post-install verification (filesystem, rocminfo, rdhc) lives only in
native_linux_package_install_test.py.

You can use this file to: import ``NativeLinuxPackageInstaller``; run the CLI for
repo-based install or for ``--simulate-packages-dir`` dry-runs; or call
``run_simulate_install_test`` from automation.

Sample use cases (Python API — run inside a matching OS container/VM, often as root):

1) Single GPU arch (DEB nightly)

    from native_linux_package_install import NativeLinuxPackageInstaller

    installer = NativeLinuxPackageInstaller(
        repo_url="https://rocm.nightlies.amd.com/deb/20260204-21658678136/",
        os_profile="ubuntu2404",
        release_type="nightly",
        gfx_arch="gfx94x",
    )
    if not installer.run_repo_setup_and_install():
        raise SystemExit(1)

2) Multiple GPU arches in one install transaction

    installer = NativeLinuxPackageInstaller(
        repo_url="https://rocm.prereleases.amd.com/packages/rhel8/x86_64/",
        os_profile="rhel8",
        release_type="prerelease",
        gpg_key_url="https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
        gfx_arch=["gfx94x", "gfx110x"],
    )
    assert installer.run_repo_setup_and_install()

3) Prerelease DEB with GPG

    installer = NativeLinuxPackageInstaller(
        repo_url="https://rocm.prereleases.amd.com/packages/ubuntu2404",
        os_profile="ubuntu2404",
        release_type="prerelease",
        gpg_key_url="https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg",
        gfx_arch="gfx94x",
    )
    installer.run_repo_setup_and_install()

This module is also a **standalone CLI** (install only, no verification):

  python3 native_linux_package_install.py --os-profile ubuntu2404 \\
    --repo-url https://rocm.nightlies.amd.com/deb/.../ \\
    --gfx-arch gfx94x --release-type nightly

For post-install checks only (no package install), use
native_linux_package_install_test.py; see that module's docstring.

Dry-run local packages (no repo install):

  python3 native_linux_package_install.py --simulate-packages-dir /path/to/pkgs \\
    --pkg-type deb
"""

import argparse
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path


def _env(key: str, default: str) -> str:
    """Return os.environ[key] if set and non-empty, else default."""
    v = os.environ.get(key, "").strip()
    return v if v else default


# --- Config: paths overridable via environment variables ---
REPO_NAME = _env("ROCM_REPO_NAME", "rocm-test")
APT_KEYRING_DIR = _env("ROCM_APT_KEYRING_DIR", "/etc/apt/keyrings")
APT_SOURCES_LIST = _env(
    "ROCM_APT_SOURCES_LIST", f"/etc/apt/sources.list.d/{REPO_NAME}.list"
)
APT_KEYRING_FILE = _env("ROCM_APT_KEYRING_FILE", "/etc/apt/keyrings/rocm.gpg")
ZYPP_REPOS_DIR = _env("ROCM_ZYPP_REPOS_DIR", "/etc/zypp/repos.d")
YUM_REPOS_DIR = _env("ROCM_YUM_REPOS_DIR", "/etc/yum.repos.d")

GPG_MKDIR_TIMEOUT_SEC = 10
GPG_KEY_TIMEOUT_SEC = 60
APT_UPDATE_TIMEOUT_SEC = 120
ZYPP_CLEAN_TIMEOUT_SEC = 60
ZYPP_REFRESH_TIMEOUT_SEC = 120
DNF_CLEAN_TIMEOUT_SEC = 60
INSTALL_TIMEOUT_SEC = 1800  # 30 minutes


def run_simulate_install_test(pkg_type: str, packages_dir: str) -> bool:
    """Dry-run install of local .deb or .rpm files (apt --simulate / rpm --test).

    Returns:
    True if the simulate command succeeded.
    """
    path = Path(packages_dir).resolve()
    if not path.is_dir():
        print(f"[FAIL] Not a directory: {packages_dir}", file=sys.stderr)
        return False

    if pkg_type == "deb":
        debs = [str(p.resolve()) for p in path.glob("*.deb")]
        if not debs:
            print(f"[FAIL] No .deb files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing DEB packages on host system for testing")
        cmd = ["apt", "install", "--simulate"] + debs
    elif pkg_type == "rpm":
        rpms = [str(p.resolve()) for p in path.glob("*.rpm")]
        if not rpms:
            print(f"[FAIL] No .rpm files found in {packages_dir}", file=sys.stderr)
            return False
        print("Simulate installing RPM packages for testing")
        cmd = ["rpm", "-Uvh", "--test", "--nodeps"] + rpms
    else:
        print(
            f"[FAIL] Unsupported pkg_type: {pkg_type}. Use 'deb' or 'rpm'.",
            file=sys.stderr,
        )
        return False

    try:
        subprocess.run(cmd, check=True)
        print("[PASS] Simulated install test completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(
            f"[FAIL] Simulated install failed with exit code {e.returncode}",
            file=sys.stderr,
        )
        return False
    except FileNotFoundError as e:
        print(f"[FAIL] Command not found: {e}", file=sys.stderr)
        return False


def _run_streaming(cmd: list[str], timeout_sec: int) -> int:
    """Run a command with streaming stdout/stderr and return its exit code.

    Lines are printed as they are produced. Raises subprocess.TimeoutExpired
    (after killing the process) or OSError on failure.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line.rstrip())
            sys.stdout.flush()
        return process.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        process.kill()
        raise


class NativeLinuxPackageInstaller:
    """Configure package-manager repos and install ROCm native packages."""

    @staticmethod
    def _derive_package_type(os_profile: str) -> str:
        """Derive package type from OS profile.

        Args:
        os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles16, almalinux9, centos7, azl3)

        Returns:
        Package type ('deb' or 'rpm')
        """
        os_profile_lower = os_profile.lower()
        if os_profile_lower.startswith(("ubuntu", "debian")):
            return "deb"
        elif os_profile_lower.startswith(
            ("rhel", "sles", "almalinux", "centos", "azl")
        ):
            return "rpm"
        else:
            raise ValueError(
                f"Unable to derive package type from OS profile: {os_profile}. "
                "Supported profiles: ubuntu*, debian*, rhel*, sles*, almalinux*, centos*, azl*"
            )

    @staticmethod
    def _normalized_gfx_archs_from_input(
        gfx_arch: str | list[str] | None,
    ) -> list[str]:
        """Normalize GPU arch list: split commas, strip, lowercase, dedupe (order kept).

        Args:
        gfx_arch: Single arch string, list of arch strings, or None (default gfx94x).

        Returns:
        Non-empty list of unique architecture tokens (e.g. ['gfx94x', 'gfx110x']).
        """
        if gfx_arch is None:
            tokens: list[str] = ["gfx94x"]
        elif isinstance(gfx_arch, str):
            tokens = [gfx_arch] if gfx_arch.strip() else ["gfx94x"]
        else:
            tokens = [str(a) for a in gfx_arch if a and str(a).strip()] or ["gfx94x"]
        expanded: list[str] = []
        for t in tokens:
            for part in t.split(","):
                p = part.strip()
                if p:
                    expanded.append(p)
        if not expanded:
            expanded = ["gfx94x"]
        seen: dict[str, None] = {}
        out: list[str] = []
        for a in expanded:
            k = a.lower()
            if k in seen:
                continue
            seen[k] = None
            out.append(k)
        return out

    def _is_sles(self) -> bool:
        """Check if the OS profile is SLES (SUSE Linux Enterprise Server).

        Returns:
        True if SLES, False otherwise
        """
        return self.os_profile.lower().startswith("sles")

    def __init__(
        self,
        repo_url: str,
        os_profile: str,
        release_type: str = "nightly",
        gfx_arch: str | list[str] | None = None,
        gpg_key_url: str | None = None,
    ):
        """Initialize the native Linux package installer.

        Args:
        repo_url: Full repository URL (constructed in YAML)
        os_profile: OS profile (e.g., ubuntu2404, rhel8, debian12, sles15, sles16, almalinux9, centos7, azl3)
        release_type: Type of release ('nightly' or 'prerelease')
        gfx_arch: GPU architecture(s) as a single value or list (default: gfx94x).
        For each architecture, installs amdrocm-{arch} and amdrocm-core-sdk-{arch}.
        gpg_key_url: GPG key URL
        """
        self.os_profile = os_profile.lower()
        self.package_type = self._derive_package_type(os_profile)
        self.repo_url = repo_url.rstrip("/")
        self.release_type = release_type.lower()
        self.gfx_arch_list = self._normalized_gfx_archs_from_input(gfx_arch)
        self.gfx_arch = self.gfx_arch_list[0]
        self.gpg_key_url = gpg_key_url

        self.package_names: list[str] = []
        for arch in self.gfx_arch_list:
            self.package_names.extend(
                [f"amdrocm-{arch}", f"amdrocm-core-sdk-{arch}"]
            )

    def setup_gpg_key(self) -> bool:
        """Setup GPG key for repositories that require GPG verification.

        Returns:
        True if setup successful, False otherwise
        """
        if not self.gpg_key_url:
            return True

        print("\n" + "=" * 80)
        print("SETTING UP GPG KEY")
        print("=" * 80)

        print(f"\nGPG Key URL: {self.gpg_key_url}")

        if self.package_type == "deb":
            keyring_dir = Path(APT_KEYRING_DIR)
            keyring_file = keyring_dir / "rocm.gpg"

            try:
                print(f"\nCreating keyring directory: {keyring_dir}...")
                subprocess.run(
                    ["mkdir", "--parents", "--mode=0755", str(keyring_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_MKDIR_TIMEOUT_SEC,
                )
                print(f"[PASS] Created keyring directory: {keyring_dir}")

                print(f"\nDownloading and importing GPG key from {self.gpg_key_url}...")
                pipeline_cmd = (
                    f"wget -q -O - {self.gpg_key_url} | "
                    f"gpg --dearmor | "
                    f"tee {keyring_file} > /dev/null"
                )

                subprocess.run(
                    pipeline_cmd,
                    shell=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=GPG_KEY_TIMEOUT_SEC,
                )

                keyring_file.chmod(0o644)
                print(f"[PASS] GPG key imported to {keyring_file}")
                return True

            except subprocess.CalledProcessError as e:
                print(f"[FAIL] Failed to setup GPG key: {e}")
                if e.stderr:
                    print(f"Error output: {e.stderr.decode()}")
                return False
            except OSError as e:
                print(f"[FAIL] Error setting up GPG key: {e}")
                return False
        else:
            return True

    def setup_deb_repository(self) -> bool:
        """Setup DEB repository on the system.

        Returns:
        True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP DEB REPOSITORY")
        print("=" * 80)

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")

        if self.gpg_key_url:
            if not self.setup_gpg_key():
                return False

        print("\nAdding ROCm repository...")
        sources_list = Path(APT_SOURCES_LIST)

        if self.gpg_key_url:
            apt_keyring = Path(APT_KEYRING_FILE)
            repo_entry = f"deb [arch=amd64 signed-by={apt_keyring}] {self.repo_url} stable main\n"
        else:
            repo_entry = f"deb [arch=amd64 trusted=yes] {self.repo_url} stable main\n"

        try:
            sources_list.write_text(repo_entry, encoding="utf-8")
            print(f"[PASS] Repository added to {sources_list}")
            print(f" {repo_entry.strip()}")
        except OSError as e:
            print(f"[FAIL] Failed to add repository: {e}")
            return False

        print("\nUpdating package lists...")
        print("=" * 80)
        try:
            return_code = _run_streaming(["apt", "update"], APT_UPDATE_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Package lists updated")
                return True
            print(f"\n[FAIL] Failed to update package lists (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] apt update timed out")
            return False
        except OSError as e:
            print(f"[FAIL] Error updating package lists: {e}")
            return False

    def _setup_sles_repository(self) -> bool:
        """Setup repository for SLES using zypper.

        Returns:
        True if setup successful, False otherwise
        """
        repo_name = REPO_NAME
        repo_file = Path(ZYPP_REPOS_DIR) / f"{repo_name}.repo"

        print(f"\nRemoving existing repository '{repo_name}' if it exists...")
        subprocess.run(
            ["zypper", "--non-interactive", "removerepo", repo_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        print(f"\nCreating ROCm repository file at {repo_file}...")
        if self.gpg_key_url:
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            repo_content = f"""[{repo_name}]
name=ROCm {self.release_type} repository
baseurl={self.repo_url}
enabled=1
gpgcheck=0
"""

        try:
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
            print(repo_content)
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        print("\nCleaning zypper cache...")
        try:
            result = subprocess.run(
                ["zypper", "--non-interactive", "clean", "--all"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=ZYPP_CLEAN_TIMEOUT_SEC,
            )
            if result.returncode == 0:
                print("[PASS] zypper cache cleaned")
            else:
                print(
                    f"[WARN] zypper clean returned {result.returncode} (may not be critical)"
                )
        except subprocess.TimeoutExpired:
            print("[WARN] zypper clean timed out (may not be critical)")
        except (subprocess.CalledProcessError, OSError) as e:
            print(f"[WARN] zypper clean failed: {e} (may not be critical)")

        print("\nRefreshing repository metadata...")
        try:
            refresh_cmd = ["zypper", "--non-interactive"]
            if self.gpg_key_url:
                refresh_cmd.append("--gpg-auto-import-keys")
            refresh_cmd.extend(["refresh", repo_name])
            return_code = _run_streaming(refresh_cmd, ZYPP_REFRESH_TIMEOUT_SEC)
            if return_code == 0:
                print("\n[PASS] Repository metadata refreshed")
                return True
            print(
                f"\n[FAIL] Failed to refresh repository metadata (exit code: {return_code})"
            )
            return False
        except subprocess.TimeoutExpired:
            print("\n[FAIL] zypper refresh timed out")
            return False
        except OSError as e:
            print(f"[FAIL] Error refreshing repository metadata: {e}")
            return False

    def _setup_dnf_repository(self) -> bool:
        """Setup repository for RHEL/AlmaLinux/CentOS using dnf/yum.

        Returns:
        True if setup successful, False otherwise
        """
        print("\nUsing dnf/yum for repository setup...")

        print("\nCreating ROCm repository file...")
        repo_name = REPO_NAME
        repo_file = Path(YUM_REPOS_DIR) / f"{repo_name}.repo"

        if self.gpg_key_url:
            repo_content = f"""[{repo_name}]
name=ROCm Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=1
gpgkey={self.gpg_key_url}
"""
        else:
            repo_content = f"""[{repo_name}]
name=Native Linux Package Test Repository
baseurl={self.repo_url}
enabled=1
gpgcheck=0
"""

        try:
            repo_file.write_text(repo_content, encoding="utf-8")
            print(f"[PASS] Repository file created: {repo_file}")
            print("\nRepository configuration:")
            print(repo_content)
        except OSError as e:
            print(f"[FAIL] Failed to create repository file: {e}")
            return False

        print("\nCleaning dnf cache...")
        try:
            subprocess.run(
                ["dnf", "clean", "all"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=DNF_CLEAN_TIMEOUT_SEC,
            )
            print("[PASS] dnf cache cleaned")
        except subprocess.CalledProcessError as e:
            print("[WARN] Failed to clean dnf cache (may not be critical)")
            print(f"Error: {e.stdout}")
        except subprocess.TimeoutExpired:
            print("[WARN] dnf clean timed out (may not be critical)")

        print("\n[PASS] DNF repository setup complete")
        return True

    def setup_rpm_repository(self) -> bool:
        """Setup RPM repository on the system.

        Returns:
        True if setup successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("SETTING UP RPM REPOSITORY")
        print("=" * 80)

        print(f"\nRepository URL: {self.repo_url}")
        print(f"Release Type: {self.release_type}")
        print(f"OS Profile: {self.os_profile}")

        if self.gpg_key_url and not self._is_sles():
            if not self.setup_gpg_key():
                return False

        if self._is_sles():
            return self._setup_sles_repository()
        else:
            return self._setup_dnf_repository()

    def install_deb_packages(self) -> bool:
        """Install ROCm DEB packages from repository.

        Returns:
        True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING DEB PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackages to install (in order): {self.package_names}")

        cmd = ["apt", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] DEB packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install DEB packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def install_rpm_packages(self) -> bool:
        """Install ROCm RPM packages from repository.

        Returns:
        True if installation successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("INSTALLING RPM PACKAGES FROM REPOSITORY")
        print("=" * 80)

        print(f"\nPackages to install (in order): {self.package_names}")

        if self._is_sles():
            if not self.gpg_key_url:
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--no-gpg-checks",
                    "install",
                    "-y",
                ] + self.package_names
            else:
                cmd = [
                    "zypper",
                    "--non-interactive",
                    "--gpg-auto-import-keys",
                    "install",
                    "-y",
                ] + self.package_names
            print("[INFO] Using zypper for SLES package installation")
        else:
            cmd = ["dnf", "install", "-y"] + self.package_names
        print(f"\nRunning: {' '.join(cmd)}")
        print("=" * 80)
        print("Installation progress (streaming output):\n")

        try:
            return_code = _run_streaming(cmd, INSTALL_TIMEOUT_SEC)
            if return_code == 0:
                print("\n" + "=" * 80)
                print("[PASS] RPM packages installed successfully from repository")
                return True
            print("\n" + "=" * 80)
            print(f"[FAIL] Failed to install RPM packages (exit code: {return_code})")
            return False
        except subprocess.TimeoutExpired:
            print("\n" + "=" * 80)
            print(f"[FAIL] Installation timed out after {INSTALL_TIMEOUT_SEC} minutes")
            return False
        except OSError as e:
            print(f"\n[FAIL] Error during installation: {e}")
            return False

    def run_repo_setup_and_install(self) -> bool:
        """Step 1: Repo setup and install.

        Returns:
        True if repository setup and package installation both succeeded.
        """
        print("\n" + "=" * 80)
        print("STEP 1: REPOSITORY SETUP AND PACKAGE INSTALLATION")
        print("=" * 80)
        print(f"\nOS Profile: {self.os_profile}")
        print(f"Package Type (derived): {self.package_type.upper()}")
        print(f"GPU Architecture(s): {self.gfx_arch_list}")
        print(f"Repository URL: {self.repo_url}")
        print(f"Packages (in order): {self.package_names}")

        if self.package_type == "deb":
            if not self.setup_deb_repository():
                return False
            return self.install_deb_packages()
        else:
            if not self.setup_rpm_repository():
                return False
            return self.install_rpm_packages()


_INSTALL_CLI_EPILOG = """
Examples:
 # Install from a nightly DEB repo (run inside matching container/VM, often as root)
 python3 native_linux_package_install.py --os-profile ubuntu2404 \\
   --repo-url https://rocm.nightlies.amd.com/deb/20260204-21658678136/ \\
   --gfx-arch gfx94x --release-type nightly

 # RPM prerelease with GPG, multiple GPU arches
 python3 native_linux_package_install.py --os-profile rhel8 \\
   --repo-url https://rocm.prereleases.amd.com/packages/rhel8/x86_64/ \\
   --gfx-arch gfx94x gfx110x --release-type prerelease \\
   --gpg-key-url https://rocm.prereleases.amd.com/packages/gpg/rocm.gpg

 # Dry-run local packages (CI package build check)
 python3 native_linux_package_install.py --simulate-packages-dir /path/to/pkgs --pkg-type deb
"""


def _build_install_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description=(
            "Configure repos and install ROCm native packages, or dry-run local .deb/.rpm. "
            "Post-install verification is native_linux_package_install_test.py only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_INSTALL_CLI_EPILOG,
    )
    parser.add_argument(
        "--simulate-packages-dir",
        metavar="DIR",
        default=None,
        help=(
            "If set, run apt/rpm dry-run on *.deb or *.rpm in this directory only "
            "(no repo install). Use --pkg-type or --os-profile to select deb vs rpm."
        ),
    )
    parser.add_argument(
        "--pkg-type",
        type=str,
        choices=["deb", "rpm"],
        default=None,
        help="Package type for --simulate-packages-dir when --os-profile is omitted.",
    )
    parser.add_argument(
        "--os-profile",
        type=str,
        default=None,
        help="OS profile (e.g. ubuntu2404, rhel8, sles16). Required for repo install; "
        "optional for simulate if --pkg-type is set.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default=None,
        help="Full repository base URL (repo install mode).",
    )
    parser.add_argument(
        "--gfx-arch",
        type=str,
        nargs="+",
        metavar="ARCH",
        default=None,
        help="GPU architecture(s); installs amdrocm-ARCH and amdrocm-core-sdk-ARCH per arch.",
    )
    parser.add_argument(
        "--release-type",
        type=str,
        choices=["dev", "nightly", "prerelease", "release", "ci"],
        default="nightly",
        help="Release channel (default: nightly).",
    )
    parser.add_argument(
        "--gpg-key-url",
        type=str,
        default=None,
        help="GPG key URL when the repo requires signed metadata.",
    )
    return parser


def parse_install_cli_arguments(argv: list[str] | None = None) -> Namespace:
    """Parse argv for the install CLI."""
    parser = _build_install_argument_parser()
    return parser.parse_args(argv)


def run_install_cli(args: Namespace) -> int:
    """Run simulate dry-run, or repo setup + package install. Returns process exit code."""
    if args.simulate_packages_dir:
        if args.pkg_type:
            pkg_type = args.pkg_type
        elif args.os_profile:
            try:
                pkg_type = NativeLinuxPackageInstaller._derive_package_type(args.os_profile)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 2
        else:
            print(
                "Error: --simulate-packages-dir requires --pkg-type or --os-profile",
                file=sys.stderr,
            )
            return 2
        print("\n" + "=" * 80)
        print("SIMULATED INSTALL (dry-run)")
        print("=" * 80)
        return 0 if run_simulate_install_test(pkg_type, args.simulate_packages_dir) else 1

    if not args.os_profile or not args.repo_url or not args.gfx_arch:
        print(
            "Error: repo install requires --os-profile, --repo-url, and --gfx-arch "
            "(or use --simulate-packages-dir for dry-run).",
            file=sys.stderr,
        )
        return 2

    try:
        NativeLinuxPackageInstaller._derive_package_type(args.os_profile)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    print("\n" + "=" * 80)
    print("NATIVE LINUX PACKAGE INSTALL (install only)")
    print("=" * 80)
    print(f"OS Profile: {args.os_profile}")
    print(f"Repository URL: {args.repo_url}")
    print(
        f"GPU Architecture(s): {args.gfx_arch} "
        f"(normalized: {NativeLinuxPackageInstaller._normalized_gfx_archs_from_input(args.gfx_arch)})"
    )
    print(f"Release Type: {args.release_type}")
    if args.gpg_key_url:
        print(f"GPG Key URL: {args.gpg_key_url}")
    print("=" * 80)

    installer = NativeLinuxPackageInstaller(
        repo_url=args.repo_url,
        os_profile=args.os_profile,
        release_type=args.release_type,
        gfx_arch=args.gfx_arch,
        gpg_key_url=args.gpg_key_url,
    )
    return 0 if installer.run_repo_setup_and_install() else 1


def main() -> None:
    """CLI entry: install from repository only."""
    args = parse_install_cli_arguments()
    sys.exit(run_install_cli(args))


if __name__ == "__main__":
    main()
