# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
import os
import sys
import subprocess
import unittest
from unittest.mock import patch

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from configure_ci_path_filters import (
    get_git_modified_paths,
    is_ci_run_required,
    _GITHUB_WORKFLOWS_CI_FILENAMES,
)
from workflow_utils import get_transitive_workflow_uses


class ConfigureCIPathFiltersTest(unittest.TestCase):
    def test_run_ci_if_source_file_edited(self):
        paths = ["source_file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_only_markdown_files_edited(self):
        paths = ["README.md", "build_tools/README.md"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_experimental_files_edited(self):
        paths = ["experimental/file.h"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_dont_run_ci_if_only_skipped_files_edited(self):
        paths = ["gitleaks.toml", "build_tools/scan_tools/script.py"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_related_workflow_file_edited(self):
        paths = [".github/workflows/multi_arch_ci.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/multi_arch_build_portable_linux_artifacts.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

        paths = [".github/workflows/multi_arch_build_native_linux_packages.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_dont_run_ci_if_unrelated_workflow_file_edited(self):
        paths = [".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

        paths = [".github/workflows/test_jax_dockerfile.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertFalse(run_ci)

    def test_run_ci_if_source_file_and_unrelated_workflow_file_edited(self):
        paths = ["source_file.h", ".github/workflows/pre-commit.yml"]
        run_ci = is_ci_run_required(paths)
        self.assertTrue(run_ci)

    def test_ci_workflow_filenames_cover_all_transitive_uses(self):
        """_GITHUB_WORKFLOWS_CI_FILENAMES must exactly match the set of
        workflows transitively called by multi_arch_ci.yml.

        This is a change-detector test that can be removed if
        _GITHUB_WORKFLOWS_CI_FILENAMES is computed dynamically instead of
        maintained by hand.

        If this test fails, update _GITHUB_WORKFLOWS_CI_FILENAMES in
        configure_ci_path_filters.py to match the actual workflow tree.
        """
        all_used = get_transitive_workflow_uses(["multi_arch_ci.yml"])
        missing = all_used - _GITHUB_WORKFLOWS_CI_FILENAMES
        stale = _GITHUB_WORKFLOWS_CI_FILENAMES - all_used
        errors = []
        if missing:
            errors.append(
                "Missing (add to _GITHUB_WORKFLOWS_CI_FILENAMES):\n"
                + "\n".join(f"  - {f}" for f in sorted(missing))
            )
        if stale:
            errors.append(
                "Stale (remove from _GITHUB_WORKFLOWS_CI_FILENAMES):\n"
                + "\n".join(f"  - {f}" for f in sorted(stale))
            )
        if errors:
            self.fail("\n".join(errors))


class GetGitModifiedPathsTest(unittest.TestCase):
    """Tests for get_git_modified_paths base-ref error handling (issue #6162)."""

    def _raise_called_process_error(self, *args, **kwargs):
        # Mimic `git diff --name-only <bad_ref>` failing with exit 128
        # ("fatal: bad object"), which is what GitHub produces for a tag
        # push (all-zeros "before" SHA) or a shallow checkout missing the
        # base commit.
        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "diff", "--name-only", "badref"],
        )

    def test_returns_none_for_all_zeros_base_ref(self):
        # Tag push: the GitHub event "before" SHA is all zeros.
        with patch("subprocess.run", side_effect=self._raise_called_process_error):
            result = get_git_modified_paths("0" * 40)
        self.assertIsNone(result)

    def test_returns_none_for_missing_base_commit(self):
        # Rebase & merge with a shallow (depth 1) checkout: the base commit
        # SHA is not present locally.
        with patch("subprocess.run", side_effect=self._raise_called_process_error):
            result = get_git_modified_paths("f5c168058a7ceaa0f179cc36784b491a11a3adc7")
        self.assertIsNone(result)

    def test_returns_none_on_timeout(self):
        with patch("subprocess.run", side_effect=TimeoutError):
            result = get_git_modified_paths("HEAD^1")
        self.assertIsNone(result)

    def test_returns_paths_on_success(self):
        fake = subprocess.CompletedProcess(
            args=["git", "diff", "--name-only", "HEAD^1"],
            returncode=0,
            stdout="a.py\nb/c.cpp\n",
        )
        with patch("subprocess.run", return_value=fake):
            result = get_git_modified_paths("HEAD^1")
        self.assertEqual(list(result), ["a.py", "b/c.cpp"])


if __name__ == "__main__":
    unittest.main()
