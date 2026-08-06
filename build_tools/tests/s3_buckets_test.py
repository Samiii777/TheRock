#!/usr/bin/env python
"""Unit tests for s3_buckets.py."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

import _therock_utils.s3_buckets as s3_buckets_module
from _therock_utils.s3_buckets import (
    CdnRule,
    get_artifacts_bucket_config,
    get_artifacts_bucket_config_for_workflow_run,
    get_release_bucket_config,
    load_bucket_config_file,
    resolve_public_url,
)


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfig(unittest.TestCase):
    def test_ci_rocm_therock(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts")
        self.assertEqual(
            config.write_access_iam_role,
            "arn:aws:iam::692859939525:role/therock-ci",
        )

    def test_ci_fork_pr(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/TheRock", is_pr_from_fork=True
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        # The raw lookup returns the external role for forks; the OIDC skip for
        # forks happens in get_artifacts_bucket_config_for_workflow_run, not here.
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_ci_external_repo(self):
        config = get_artifacts_bucket_config(
            release_type="ci", repository="ROCm/rocm-libraries", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_release_type_dev(self):
        config = get_artifacts_bucket_config(
            release_type="dev", repository="ROCm/TheRock", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-dev-artifacts")
        self.assertEqual(config.iam_role, "therock-dev")

    def test_release_type_from_rockrel(self):
        config = get_artifacts_bucket_config(
            release_type="nightly", repository="ROCm/rockrel", is_pr_from_fork=False
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_invalid_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_artifacts_bucket_config(
                release_type="bogus",
                repository="ROCm/TheRock",
                is_pr_from_fork=False,
            )
        self.assertIn("bogus", str(cm.exception))

    def test_empty_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_artifacts_bucket_config(
                release_type="",
                repository="ROCm/TheRock",
                is_pr_from_fork=False,
            )


# ---------------------------------------------------------------------------
# get_release_bucket_config
# ---------------------------------------------------------------------------


class TestGetReleaseBucketConfig(unittest.TestCase):
    def test_dev_tarball(self):
        config = get_release_bucket_config(release_type="dev", bucket_type="tarball")
        self.assertEqual(config.name, "therock-dev-tarball")
        self.assertEqual(config.iam_role, "therock-dev")
        self.assertEqual(
            config.write_access_iam_role,
            "arn:aws:iam::692859939525:role/therock-dev",
        )

    def test_nightly_python(self):
        config = get_release_bucket_config(release_type="nightly", bucket_type="python")
        self.assertEqual(config.name, "therock-nightly-python")
        self.assertEqual(config.iam_role, "therock-nightly")

    def test_prerelease_packages(self):
        config = get_release_bucket_config(
            release_type="prerelease", bucket_type="packages"
        )
        self.assertEqual(config.name, "therock-prerelease-packages")
        self.assertEqual(config.iam_role, "therock-prerelease")

    def test_all_combinations_exist(self):
        for release_type in ("dev", "nightly", "prerelease"):
            for bucket_type in ("tarball", "python", "packages"):
                config = get_release_bucket_config(release_type, bucket_type)
                self.assertEqual(config.name, f"therock-{release_type}-{bucket_type}")

    def test_invalid_release_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_release_bucket_config(release_type="bogus", bucket_type="tarball")
        self.assertIn("bogus", str(cm.exception))

    def test_empty_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_release_bucket_config(release_type="", bucket_type="tarball")

    def test_ci_release_type_raises(self):
        with self.assertRaises(ValueError):
            get_release_bucket_config(release_type="ci", bucket_type="tarball")

    def test_invalid_bucket_type_raises(self):
        with self.assertRaises(ValueError) as cm:
            get_release_bucket_config(release_type="dev", bucket_type="wheels")
        self.assertIn("wheels", str(cm.exception))


# ---------------------------------------------------------------------------
# get_artifacts_bucket_config_for_workflow_run
# ---------------------------------------------------------------------------


class TestGetArtifactsBucketConfigForWorkflowRun(unittest.TestCase):
    """Test the workflow-run-aware wrapper."""

    def setUp(self):
        self.api_patcher = mock.patch(
            "github_actions.github_actions_api.gha_query_workflow_run_by_id"
        )
        self.mock_api = self.api_patcher.start()

        self.env_patcher = mock.patch.dict(os.environ)
        self.env_patcher.start()
        os.environ.pop("GITHUB_REPOSITORY", None)
        os.environ.pop("GITHUB_EVENT_NAME", None)
        os.environ.pop("GITHUB_EVENT_PATH", None)
        os.environ.pop("RELEASE_TYPE", None)

    def tearDown(self):
        self.env_patcher.stop()
        self.api_patcher.stop()

    def test_default_ci(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_explicit_release_type(self):
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_release_type_from_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock"
        )
        self.assertEqual(config.name, "therock-dev-artifacts")

    def test_explicit_release_type_overrides_env(self):
        os.environ["RELEASE_TYPE"] = "dev"
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", release_type="nightly"
        )
        self.assertEqual(config.name, "therock-nightly-artifacts")

    def test_workflow_run_same_repo(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts")

    def test_workflow_run_from_fork(self):
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "SomeUser/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        # Fork PRs cannot assume an IAM role via OIDC (no trust relationship),
        # so the wrapper must strip the role and fall back to runner base
        # credentials. Regression coverage for #5654.
        self.assertIsNone(config.iam_role)
        self.assertIsNone(config.write_access_iam_role)

    def test_workflow_run_external_repo_uses_oidc(self):
        # An external (non-fork) repo such as rocm-libraries keeps the
        # therock-ci-external role so it can authenticate via OIDC.
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/rocm-libraries"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/rocm-libraries", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts-external")
        self.assertEqual(config.iam_role, "therock-ci-external")

    def test_workflow_run_same_repo_keeps_internal_role(self):
        # A same-repo (non-fork) ROCm/TheRock PR is unaffected by the fork skip
        # and keeps the internal therock-ci role.
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run=fake_run
        )
        self.assertEqual(config.name, "therock-ci-artifacts")
        self.assertEqual(config.iam_role, "therock-ci")

    def test_workflow_run_id_triggers_api_call(self):
        self.mock_api.return_value = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock", workflow_run_id="12345"
        )
        self.mock_api.assert_called_once_with("ROCm/TheRock", "12345")
        self.assertEqual(config.name, "therock-ci-artifacts")

    def _write_event(self, event: dict) -> str:
        """Write a synthetic GitHub event payload to a temp file.

        Returns the path. Caller must os.unlink() after use.
        Uses delete=False because NamedTemporaryFile(delete=True) holds an
        exclusive lock on Windows, preventing the code under test from reading.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(event, f)
            return f.name

    def test_fork_pr_from_event_payload(self):
        """Fork PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": True}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts-external")
        finally:
            os.unlink(event_path)

    def test_same_repo_pr_from_event_payload(self):
        """Same-repo PR detected via event payload (no workflow_run dict)."""
        event_path = self._write_event(
            {"pull_request": {"head": {"repo": {"fork": False}}}}
        )
        try:
            os.environ["GITHUB_EVENT_NAME"] = "pull_request"
            os.environ["GITHUB_EVENT_PATH"] = event_path
            config = get_artifacts_bucket_config_for_workflow_run(
                github_repository="ROCm/TheRock"
            )
            self.assertEqual(config.name, "therock-ci-artifacts")
        finally:
            os.unlink(event_path)

    def test_workflow_run_id_ignored_when_workflow_run_provided(self):
        """workflow_run takes priority over workflow_run_id."""
        fake_run = {
            "id": 12345,
            "head_repository": {"full_name": "ROCm/TheRock"},
        }
        config = get_artifacts_bucket_config_for_workflow_run(
            github_repository="ROCm/TheRock",
            workflow_run=fake_run,
            workflow_run_id="99999",
        )
        self.mock_api.assert_not_called()
        self.assertEqual(config.name, "therock-ci-artifacts")


# ---------------------------------------------------------------------------
# CdnRule and public URL resolution
# ---------------------------------------------------------------------------


class TestCdnRule(unittest.TestCase):
    def test_key_prefix_must_end_with_slash(self):
        with self.assertRaises(ValueError):
            CdnRule("deb", "https://cdn.example.com/")

    def test_empty_key_prefix_is_allowed(self):
        rule = CdnRule("", "https://cdn.example.com/whl/")
        self.assertEqual(rule.key_prefix, "")

    def test_url_prefix_must_be_https(self):
        with self.assertRaises(ValueError):
            CdnRule("", "http://cdn.example.com/")

    def test_url_prefix_trailing_slash_normalized(self):
        rule = CdnRule("", "https://cdn.example.com/whl/")
        self.assertEqual(rule.url_prefix, "https://cdn.example.com/whl")


class TestResolvePublicUrl(unittest.TestCase):
    def test_bucket_wide_cdn(self):
        self.assertEqual(
            resolve_public_url("therock-nightly-python", "a/b.whl"),
            "https://rocm.nightlies.amd.com/whl-multi-arch/a/b.whl",
        )

    def test_longest_prefix_wins(self):
        self.assertEqual(
            resolve_public_url("therock-nightly-packages", "deb/pool/x.deb"),
            "https://rocm.nightlies.amd.com/packages-multi-arch/deb/pool/x.deb",
        )
        self.assertEqual(
            resolve_public_url("therock-nightly-packages", "rpm/el9/x.rpm"),
            "https://rocm.nightlies.amd.com/packages-multi-arch/rpm/el9/x.rpm",
        )

    def test_no_matching_rule_falls_back_to_raw_s3(self):
        self.assertEqual(
            resolve_public_url("therock-nightly-packages", "misc/x"),
            "https://therock-nightly-packages.s3.amazonaws.com/misc/x",
        )

    def test_unknown_bucket_falls_back_to_raw_s3(self):
        self.assertEqual(
            resolve_public_url("therock-ci-artifacts", "a/b"),
            "https://therock-ci-artifacts.s3.amazonaws.com/a/b",
        )


# ---------------------------------------------------------------------------
# JSON bucket registry file
# ---------------------------------------------------------------------------


class TestLoadBucketConfigFile(unittest.TestCase):
    def setUp(self):
        self._saved = dict(s3_buckets_module._BUCKET_CONFIGS_BY_NAME)
        self.addCleanup(self._restore)

    def _restore(self):
        s3_buckets_module._BUCKET_CONFIGS_BY_NAME.clear()
        s3_buckets_module._BUCKET_CONFIGS_BY_NAME.update(self._saved)

    def _write(self, entries):
        f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump(entries, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_additive_merge_new_bucket(self):
        path = self._write(
            [
                {
                    "name": "my-org-artifacts",
                    "cdn_rules": [
                        {"key_prefix": "", "url_prefix": "https://cdn.my-org.com/"}
                    ],
                }
            ]
        )
        load_bucket_config_file(path)
        self.assertEqual(
            resolve_public_url("my-org-artifacts", "x/y"),
            "https://cdn.my-org.com/x/y",
        )

    def test_override_requires_explicit_flag(self):
        path = self._write([{"name": "therock-nightly-python", "cdn_rules": []}])
        with self.assertRaises(ValueError):
            load_bucket_config_file(path)

    def test_override_true_shadows_builtin(self):
        path = self._write(
            [
                {
                    "name": "therock-nightly-python",
                    "override": True,
                    "cdn_rules": [
                        {"key_prefix": "", "url_prefix": "https://new.example.com/"}
                    ],
                }
            ]
        )
        load_bucket_config_file(path)
        self.assertEqual(
            resolve_public_url("therock-nightly-python", "a"),
            "https://new.example.com/a",
        )

    def test_unknown_key_is_error(self):
        path = self._write([{"name": "zzz", "bogus": 1}])
        with self.assertRaises(ValueError):
            load_bucket_config_file(path)


if __name__ == "__main__":
    unittest.main()
