# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Installation package tests for the core package."""

import importlib
import locale
from pathlib import Path
import platform
import subprocess
import sys
import unittest

from .. import _dist_info as di
from . import utils

import rocm_sdk

utils.assert_is_physical_package(rocm_sdk)

core_mod_name = di.ALL_PACKAGES["core"].get_py_package_name()
core_mod = importlib.import_module(core_mod_name)
utils.assert_is_physical_package(core_mod)

so_paths = utils.get_module_shared_libraries(core_mod)
is_windows = platform.system() == "Windows"

# Console script tests are templated across tuples of
#   (script_name, cl, expected_text, required)
# For example:
#   ("hipcc", ["--help"], "clang LLVM compiler", True)
#   This will run `hipcc --help` and check the output for "clang LLVM compiler".
#   If the script does not exist, the test case will fail.
COMMON_CONSOLE_SCRIPT_TESTS = [
    ("amdclang", ["--help"], "clang LLVM compiler", True),
    ("amdclang++", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cpp", ["--help"], "clang LLVM compiler", True),
    ("amdclang-cl", ["-help"], "clang LLVM compiler", True),
    ("amdflang", ["--help"], "LLVM compiler", True),
    ("amdlld", ["-flavor", "ld.lld", "--help"], "USAGE:", True),
    ("hipcc", ["--help"], "clang LLVM compiler", True),
    ("hipconfig", [], "HIP version:", True),
    ("hipify-clang", ["--help"], "USAGE:", True),
]

LINUX_CONSOLE_SCRIPT_TESTS = [
    ("amd-smi", [], "AMD-SMI", True),
    ("rocm_agent_enumerator", [], "", True),
    ("rocminfo", [], "", True),
    ("rocm-smi", [], "Management", True),
    ("hipify-perl", ["--help"], "USAGE:", True),
]

WINDOWS_CONSOLE_SCRIPT_TESTS = [
    ("hipInfo", [], "", True),
]

CONSOLE_SCRIPT_TESTS = COMMON_CONSOLE_SCRIPT_TESTS + (
    WINDOWS_CONSOLE_SCRIPT_TESTS if is_windows else LINUX_CONSOLE_SCRIPT_TESTS
)


# System (non-bundled) runtime libraries that ROCm shared objects link against
# but that TheRock does not vendor. If one of these is missing, dlopen fails
# with "cannot open shared object file" and the user must install the system
# package. Map the SONAME to a hint about which package provides it so the test
# can emit an actionable message instead of a bare traceback.
# See dockerfiles/install_rocm_deps.sh and docs/development/dependencies.md.
_SYSTEM_LIBRARY_PACKAGE_HINTS = {
    "libatomic.so.1": (
        "libatomic (provided by the C/C++ runtime). "
        "Install it with one of: "
        "`apt-get install libatomic1` (Debian/Ubuntu), "
        "`dnf install libatomic` (RHEL/AlmaLinux/Azure Linux), or "
        "`zypper install libatomic1` (SLES)."
    ),
}


def _format_shared_library_load_failure(so_path, output: str) -> str:
    """Builds an actionable failure message for a shared library that failed
    to load.

    If the failure is caused by a missing *system* shared object (e.g.
    libatomic.so.1 is not installed), the message names the missing library and
    the package that provides it. Otherwise the raw loader output is returned.
    """
    for soname, hint in _SYSTEM_LIBRARY_PACKAGE_HINTS.items():
        if f"{soname}: cannot open shared object file" in output:
            return (
                f"Failed to load {so_path}: missing system library {soname}.\n"
                f"This library is a runtime dependency that TheRock does not "
                f"bundle; it must be installed from your OS distribution.\n"
                f"To resolve: install {hint}\n\n"
                f"Loader output:\n{output}"
            )
    return f"Failed to load shared library {so_path}:\n{output}"


class ROCmCoreTest(unittest.TestCase):
    def testInstallationLayout(self):
        """The `rocm_sdk` and core module must be siblings on disk."""
        sdk_path = Path(rocm_sdk.__file__)
        self.assertEqual(
            sdk_path.name,
            "__init__.py",
            msg="Expected `rocm_sdk` module to be a non-namespace package",
        )
        core_path = Path(core_mod.__file__)
        self.assertEqual(
            core_path.name,
            "__init__.py",
            msg="Expected core module to be a non-namespace package",
        )
        self.assertEqual(
            sdk_path.parent.parent,
            core_path.parent.parent,
            msg="Paths are not siblings",
        )

    def testSharedLibrariesLoad(self):
        self.assertTrue(
            so_paths, msg="Expected core package to contain shared libraries"
        )

        for so_path in so_paths:
            if "amd_smi" in str(so_path) or "goamdsmi" in str(so_path):
                # TODO: Library preloads for amdsmi need to be implement.
                # Though this is not needed for the amd-smi client.
                continue
            if "clang_rt" in str(so_path):
                # clang_rt and sanitizer libraries are not all intended to be
                # loadable arbitrarily.
                continue
            if "libLLVMOffload" in str(so_path):
                # recent addition from upstream, issue tracked in
                # https://github.com/ROCm/TheRock/issues/2537
                continue
            if "lib/roctracer" in str(so_path) or "share/roctracer" in str(so_path):
                # Internal roctracer libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            if (
                "lib/rocprofiler-sdk/" in str(so_path)
                or "libexec/rocprofiler-sdk/" in str(so_path)
                or "libpyrocpd" in str(so_path)
                or "libpyroctx" in str(so_path)
            ):
                # Internal rocprofiler-sdk libraries are meant to be pre-loaded
                # explicitly and cannot necessarily be loaded standalone.
                continue
            if "libtest_linking_lib" in str(so_path):
                # rocprim unit tests, not actual library files
                continue
            if "opencl" in str(so_path):
                # We use OpenCL ICD from distro rather than TheRock
                # and we do not build it
                continue
            with self.subTest(msg="Check shared library loads", so_path=so_path):
                # Load each in an isolated process because not all libraries in the tree
                # are designed to load into the same process (i.e. LLVM runtime libs,
                # etc).
                command = "import ctypes; import sys; ctypes.CDLL(sys.argv[1])"
                cmd = [sys.executable, "-c", command, str(so_path)]
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode != 0:
                    output = result.stdout.decode(
                        locale.getpreferredencoding(), errors="replace"
                    )
                    self.fail(
                        _format_shared_library_load_failure(so_path, output)
                    )

    def testConsoleScripts(self):
        for script_name, cl, expected_text, required in CONSOLE_SCRIPT_TESTS:
            script_path = utils.find_console_script(script_name)
            if not required and script_path is None:
                continue
            with self.subTest(msg=f"Check console-script {script_name}"):
                self.assertIsNotNone(
                    script_path,
                    msg=f"Console script {script_path} does not exist",
                )
                encoding = locale.getpreferredencoding()
                output_text = subprocess.check_output(
                    [script_path] + cl,
                    stderr=subprocess.STDOUT,
                ).decode(encoding)
                if expected_text not in output_text:
                    self.fail(
                        f"Expected '{expected_text}' in console-script {script_name} outuput:\n"
                        f"{output_text}"
                    )

    def testPreloadLibraries(self):
        target_family = di.determine_target_family()

        for lib_entry in di.ALL_LIBRARIES.values():
            # Only test for packages we have installed.
            if lib_entry.package.has_py_package(target_family):
                with self.subTest(
                    msg="Check rocm_sdk.preload_libraries",
                    shortname=lib_entry.shortname,
                ):
                    rocm_sdk.preload_libraries(lib_entry.shortname)
