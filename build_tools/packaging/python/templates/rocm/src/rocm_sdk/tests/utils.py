# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Test utilities."""

from pathlib import Path
import platform
import shlex
import subprocess
import sys
import sysconfig

is_windows = platform.system() == "Windows"
exe_suffix = ".exe" if is_windows else ""


def run_command(args: list[str | Path], cwd: Path | None = None, capture: bool = False):
    args = [str(arg) for arg in args]
    if cwd is None:
        cwd = Path.cwd()
    print(f"++ Exec [{cwd}]$ {shlex.join(args)}")
    sys.stdout.flush()
    if capture:
        return subprocess.check_output(args, cwd=str(cwd), stdin=subprocess.DEVNULL)
    else:
        subprocess.check_call(args, cwd=str(cwd), stdin=subprocess.DEVNULL)


def assert_is_physical_package(mod):
    """Asserts that the given module is a non namespace module on disk defined
    by an __init__.py file."""
    assert (
        mod.__file__ is not None
    ), f"The `{mod.__name__}` module does not exist as a physical directory (__file__ is None)"
    assert (
        Path(mod.__file__).name == "__init__.py"
    ), f"Expected `{mod.__name__}` to be a non-namespace package"


def get_module_shared_libraries(mod) -> list[Path]:
    path = Path(mod.__file__).parent
    if is_windows:
        so_paths = [p for p in path.glob("**/*.dll") if p.is_file()]
    else:
        so_paths = [p for p in path.glob("**/*.so.*") if p.is_file()] + [
            p for p in path.glob("**/*.so") if p.is_file()
        ]

    return so_paths


def find_console_script(script_name: str) -> Path | None:
    scripts_paths = [sysconfig.get_path("scripts")]
    if is_windows:
        scripts_paths.append(sysconfig.get_path("scripts", "nt_user"))
    else:
        scripts_paths.append(sysconfig.get_path("scripts", "posix_user"))
    for scripts_path in scripts_paths:
        script_path = (Path(scripts_path) / script_name).with_suffix(exe_suffix)
        if script_path.exists():
            return script_path
    return None


# Maps the package-manager command for installing system packages on common
# Linux distributions. Used to produce actionable diagnostics when a bundled
# ROCm shared library fails to load because of a missing *system* dependency
# (i.e. a library that is expected to be provided by the host distro and is not
# vendored into the wheels). See https://github.com/ROCm/TheRock/issues/5819.
_SYSTEM_DEP_INSTALL_HINTS = {
    "libatomic.so.1": (
        "Debian/Ubuntu: sudo apt-get install libatomic1\n"
        "        RHEL/AlmaLinux/Fedora/Azure Linux: sudo dnf install libatomic\n"
        "        SLES/openSUSE: sudo zypper install libatomic1"
    ),
}

# Pattern emitted by the dynamic loader / ctypes when a NEEDED shared library
# cannot be found, e.g.:
#   OSError: libatomic.so.1: cannot open shared object file: No such file or directory
_MISSING_LIBRARY_MARKER = "cannot open shared object file"


def _extract_missing_library(stderr: str) -> str | None:
    """Returns the soname of the missing system library from loader stderr, if any."""
    for line in stderr.splitlines():
        if _MISSING_LIBRARY_MARKER in line:
            # The soname is the token immediately before the marker, e.g.
            #   libatomic.so.1: cannot open shared object file: ...
            candidate = line.split(_MISSING_LIBRARY_MARKER, 1)[0].strip()
            candidate = candidate.rstrip(":").strip()
            # Keep only the trailing soname token (handles "OSError: lib...:" prefixes).
            return candidate.split()[-1] if candidate else None
    return None


def format_shared_library_load_error(so_path, stderr: str) -> str:
    """Builds an actionable error message for a failed shared library load.

    When the underlying cause is a missing *system* library that ROCm expects
    the host distro to provide (most commonly ``libatomic.so.1``), the bare
    ``OSError`` is confusing because ``rocminfo`` and other tools may still
    work. This converts it into a clear, actionable message telling the user
    which system package to install.
    """
    stderr = stderr or ""
    missing = _extract_missing_library(stderr)
    if missing:
        hint = _SYSTEM_DEP_INSTALL_HINTS.get(missing)
        message = (
            f"Failed to load shared library:\n  {so_path}\n"
            f"because the system library '{missing}' is not installed.\n\n"
            "This library is provided by your operating system (it is not "
            "bundled with the ROCm Python packages) and must be installed "
            "separately."
        )
        if hint:
            message += f"\n\nInstall it with:\n        {hint}"
        message += f"\n\nUnderlying loader error:\n{stderr.strip()}"
        return message
    return f"Failed to load shared library:\n  {so_path}\n\n{stderr.strip()}"
