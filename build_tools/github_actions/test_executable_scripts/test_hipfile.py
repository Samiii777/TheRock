# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Runs hipFile's unit tests from the installed/packaged artifact tree.
#
# hipFile installs a relocatable ctest tree at share/hipfile/test (a top-level
# CTestTestfile.cmake plus script/hipfile_discover.cmake). Discovery and the
# runtime library path for hipFile's own libraries are handled inside that
# tree relative to its own location, so pointing ctest at it is enough for
# release builds.
#
# ASAN builds are the exception: the test binaries are instrumented and
# dynamically linked against the compiler runtime (libclang_rt.asan-x86_64.so),
# which is installed by the amd-llvm_lib artifact under
# <install>/lib/llvm/lib/clang/<version>/lib/linux/. That directory is not on
# the default loader search path, so ctest's discovery phase
# (internal_tests --gtest_list_tests) fails at load time with
#   "error while loading shared libraries: libclang_rt.asan-x86_64.so:
#    cannot open shared object file: No such file or directory"
# To fix this, prepend the clang runtime directory to LD_LIBRARY_PATH for ASAN
# builds so the loader can resolve the sanitizer runtime.
#
# Only the "unit" label is run: the "system" tests need a real GPU and the
# "stress" tests are gdb-wrapped concurrency testers, both excluded from the
# packaged unit suite.

import logging
import shlex
import subprocess
from pathlib import Path
import os

logging.basicConfig(level=logging.INFO)

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

if THEROCK_BIN_DIR is None:
    logging.error("env(THEROCK_BIN_DIR) is not set. Set it before running tests.")
    raise SystemExit(1)

# THEROCK_BIN_DIR is <install>/bin; the relocatable test tree is alongside it.
INSTALL_DIR = Path(THEROCK_BIN_DIR).resolve().parent
HIPFILE_TEST_DIR = INSTALL_DIR / "share" / "hipfile" / "test"

if not HIPFILE_TEST_DIR.is_dir():
    logging.error(f"hipFile test directory not found: {HIPFILE_TEST_DIR}")
    raise SystemExit(1)


def _is_asan_build() -> bool:
    """True when running against an ASAN-instrumented build."""
    return "asan" in os.getenv("BUILD_VARIANT", "").lower()


def _find_clang_runtime_dirs() -> list[Path]:
    """Locate clang compiler-runtime directories holding the sanitizer runtime.

    The clang major-version subdirectory is not known ahead of time, so glob
    for it under <install>/lib/llvm/lib/clang/<version>/lib/linux/ (where the
    amd-llvm_lib artifact installs libclang_rt.asan-x86_64.so).
    """
    llvm_clang_dir = INSTALL_DIR / "lib" / "llvm" / "lib" / "clang"
    if not llvm_clang_dir.is_dir():
        return []
    return [d for d in sorted(llvm_clang_dir.glob("*/lib/linux")) if d.is_dir()]


# In ASAN builds the instrumented test binaries are dynamically linked against
# libclang_rt.asan-x86_64.so, which lives in a clang runtime directory that is
# not on the default loader search path. Add it to LD_LIBRARY_PATH so ctest's
# discovery phase (internal_tests --gtest_list_tests) can load the binary.
env = os.environ.copy()
if _is_asan_build():
    runtime_dirs = _find_clang_runtime_dirs()
    if runtime_dirs:
        extra = os.pathsep.join(str(d) for d in runtime_dirs)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = extra + (os.pathsep + existing if existing else "")
        logging.info(
            f"ASAN build detected: prepended clang runtime dir(s) to "
            f"LD_LIBRARY_PATH: {extra}"
        )
    else:
        logging.warning(
            "ASAN build detected but no clang runtime directory "
            "(lib/llvm/lib/clang/*/lib/linux) was found under "
            f"{INSTALL_DIR}; the sanitizer runtime may fail to load."
        )

cmd = [
    "ctest",
    "--test-dir",
    str(HIPFILE_TEST_DIR),
    "-L",
    "unit",
    "--output-on-failure",
    "--no-tests=error",
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    env=env,
    check=True,
)
