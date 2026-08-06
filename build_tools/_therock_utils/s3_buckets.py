# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inventory of S3 buckets used by CI/CD systems and related functions.

See docs/development/s3_buckets.md.
"""

from dataclasses import dataclass, field, replace
import json
import os
import sys


def _log(*args, **kwargs):
    """Log to stdout with flush for CI visibility."""
    print(*args, **kwargs)
    sys.stdout.flush()


@dataclass(frozen=True)
class CdnRule:
    """Maps an S3 key prefix to a public URL prefix (CloudFront or other CDN).

    A bucket may serve objects under a CDN at a URL layout that differs from the
    raw S3 key layout. Each rule rewrites keys beginning with ``key_prefix`` to
    ``url_prefix``. A rule with an empty ``key_prefix`` applies bucket-wide;
    when several rules match a key the longest ``key_prefix`` wins.
    """

    key_prefix: str
    """S3 key prefix this rule applies to ('' matches the whole bucket)."""

    url_prefix: str
    """Public URL prefix that ``key_prefix`` maps to (without a trailing slash)."""

    def __post_init__(self):
        if self.key_prefix and not self.key_prefix.endswith("/"):
            raise ValueError(
                f"CdnRule key_prefix={self.key_prefix!r} must end with '/'"
            )
        if not self.url_prefix.startswith("https://"):
            raise ValueError(
                f"CdnRule url_prefix={self.url_prefix!r} must start with 'https://'"
            )
        object.__setattr__(self, "url_prefix", self.url_prefix.rstrip("/"))


@dataclass(frozen=True)
class S3BucketConfig:
    """Metadata for a single bucket in S3"""

    name: str
    """S3 bucket name (e.g. 'therock-ci-artifacts')"""

    region: str = field(default="us-east-2")
    """Region in S3 (e.g. 'us-east-2')"""

    iam_account: str | None = field(default="692859939525")
    """IAM account for write_access_iam_role"""

    iam_role: str | None = field(default=None)
    """IAM role name that grants write access to this bucket (e.g. 'therock-ci'), if any"""

    key_prefix: str = field(default="")
    """Common key prefix under which this bucket stores objects (e.g. 'v4/')."""

    cdn_rules: tuple[CdnRule, ...] = field(default=())
    """Prefix-specific public URL rules (see :func:`resolve_public_url`)."""

    namespace_external_repos: bool = field(default=False)
    """Whether objects from external repos live under a per-repo key namespace."""

    @property
    def write_access_iam_role(self) -> str | None:
        """IAM role granting write access to the bucket"""
        if not self.iam_role:
            return None
        if not self.iam_account:
            raise ValueError(
                f"Bucket {self.name!r} has iam_role={self.iam_role!r} but no iam_account"
            )
        return f"arn:aws:iam::{self.iam_account}:role/{self.iam_role}"

    def public_url(self, relative_path: str) -> str:
        """Resolve ``relative_path`` to a public URL via the longest matching CDN rule.

        Falls back to the raw S3 HTTPS URL when no CDN rule matches.
        """
        best: CdnRule | None = None
        for rule in self.cdn_rules:
            if relative_path.startswith(rule.key_prefix):
                if best is None or len(rule.key_prefix) > len(best.key_prefix):
                    best = rule
        if best is None:
            return f"https://{self.name}.s3.amazonaws.com/{relative_path}"
        suffix = relative_path[len(best.key_prefix) :]
        return f"{best.url_prefix}/{suffix}"


s3_bucket_configs = [
    # CI (external repos use OIDC with therock-ci-external; fork PRs use runner base credentials)
    S3BucketConfig("therock-ci-artifacts", iam_role="therock-ci"),
    S3BucketConfig("therock-ci-artifacts-external", iam_role="therock-ci-external"),
    # Release type "dev"
    S3BucketConfig("therock-dev-artifacts", iam_role="therock-dev"),
    S3BucketConfig(
        "therock-dev-packages",
        iam_role="therock-dev",
        cdn_rules=(
            CdnRule(
                "deb/", "https://rocm.devreleases.amd.com/packages-multi-arch/deb/"
            ),
            CdnRule(
                "rpm/", "https://rocm.devreleases.amd.com/packages-multi-arch/rpm/"
            ),
        ),
    ),
    S3BucketConfig(
        "therock-dev-python",
        iam_role="therock-dev",
        cdn_rules=(CdnRule("", "https://rocm.devreleases.amd.com/whl-multi-arch/"),),
    ),
    S3BucketConfig(
        "therock-dev-tarball",
        iam_role="therock-dev",
        cdn_rules=(
            CdnRule("", "https://rocm.devreleases.amd.com/tarball-multi-arch/"),
        ),
    ),
    # Release type "nightly"
    S3BucketConfig("therock-nightly-artifacts", iam_role="therock-nightly"),
    S3BucketConfig(
        "therock-nightly-packages",
        iam_role="therock-nightly",
        cdn_rules=(
            CdnRule("deb/", "https://rocm.nightlies.amd.com/packages-multi-arch/deb/"),
            CdnRule("rpm/", "https://rocm.nightlies.amd.com/packages-multi-arch/rpm/"),
        ),
    ),
    S3BucketConfig(
        "therock-nightly-python",
        iam_role="therock-nightly",
        cdn_rules=(CdnRule("", "https://rocm.nightlies.amd.com/whl-multi-arch/"),),
    ),
    S3BucketConfig(
        "therock-nightly-tarball",
        iam_role="therock-nightly",
        cdn_rules=(CdnRule("", "https://rocm.nightlies.amd.com/tarball-multi-arch/"),),
    ),
    # Release type "prerelease"
    S3BucketConfig("therock-prerelease-artifacts", iam_role="therock-prerelease"),
    S3BucketConfig(
        "therock-prerelease-packages",
        iam_role="therock-prerelease",
        cdn_rules=(
            CdnRule("", "https://rocm.prereleases.amd.com/packages-multi-arch/"),
        ),
    ),
    S3BucketConfig(
        "therock-prerelease-python",
        iam_role="therock-prerelease",
        cdn_rules=(CdnRule("", "https://rocm.prereleases.amd.com/whl-multi-arch/"),),
    ),
    S3BucketConfig(
        "therock-prerelease-tarball",
        iam_role="therock-prerelease",
        cdn_rules=(
            CdnRule("", "https://rocm.prereleases.amd.com/tarball-multi-arch/"),
        ),
    ),
    # Release type "release" (no automated credentials for uploading)
    S3BucketConfig("therock-release-artifacts", iam_role=None),
    S3BucketConfig(
        "therock-release-packages",
        iam_role=None,
        cdn_rules=(CdnRule("", "https://repo.amd.com/rocm/packages-multi-arch/"),),
    ),
    S3BucketConfig(
        "therock-release-python",
        iam_role=None,
        cdn_rules=(CdnRule("", "https://repo.amd.com/rocm/whl-multi-arch/"),),
    ),
    S3BucketConfig(
        "therock-release-tarball",
        iam_role=None,
        cdn_rules=(CdnRule("", "https://repo.amd.com/rocm/tarball-multi-arch/"),),
    ),
]


_BUCKET_CONFIGS_BY_NAME = {c.name: c for c in s3_bucket_configs}


_KNOWN_BUCKET_KEYS = {
    "name",
    "region",
    "iam_account",
    "iam_role",
    "key_prefix",
    "cdn_rules",
    "namespace_external_repos",
    "override",
}


def _bucket_config_from_dict(entry: dict) -> S3BucketConfig:
    unknown = set(entry) - _KNOWN_BUCKET_KEYS
    if unknown:
        raise ValueError(f"Unknown bucket config keys: {sorted(unknown)}")
    cdn_rules = tuple(
        CdnRule(r["key_prefix"], r["url_prefix"]) for r in entry.get("cdn_rules", [])
    )
    return S3BucketConfig(
        name=entry["name"],
        region=entry.get("region", "us-east-2"),
        iam_account=entry.get("iam_account", "692859939525"),
        iam_role=entry.get("iam_role"),
        key_prefix=entry.get("key_prefix", ""),
        cdn_rules=cdn_rules,
        namespace_external_repos=entry.get("namespace_external_repos", False),
    )


def load_bucket_config_file(path) -> None:
    """Merge extra bucket configs from a JSON registry file into the registry.

    The additive merge lets a repository that reuses these build tools register
    its own buckets. Shadowing a built-in bucket requires an explicit
    ``"override": true`` (logged); unknown keys are errors.
    """
    with open(path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(
            f"Bucket config file {path} must be a JSON list of bucket objects"
        )
    for entry in entries:
        override = bool(entry.get("override", False))
        config = _bucket_config_from_dict(entry)
        existing = _BUCKET_CONFIGS_BY_NAME.get(config.name)
        if existing is not None and not override:
            raise ValueError(
                f'Bucket {config.name!r} already exists; set "override": true '
                f"to shadow the built-in definition"
            )
        if existing is not None:
            _log(f"Overriding built-in bucket config {config.name!r} from {path}")
        _BUCKET_CONFIGS_BY_NAME[config.name] = config


def maybe_load_bucket_config_from_env() -> None:
    """Load the registry file named by THEROCK_S3_BUCKETS_FILE, if set."""
    path = os.environ.get("THEROCK_S3_BUCKETS_FILE")
    if path:
        load_bucket_config_file(path)


def get_bucket_config(bucket_name: str) -> S3BucketConfig:
    """Look up a bucket config by name, or raise KeyError."""
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def resolve_public_url(bucket_name: str, relative_path: str) -> str:
    """Resolve a public (CDN-aware) URL for an object in a known bucket.

    Falls back to the raw S3 HTTPS URL when the bucket is unknown or has no
    matching CDN rule.
    """
    config = _BUCKET_CONFIGS_BY_NAME.get(bucket_name)
    if config is None:
        return f"https://{bucket_name}.s3.amazonaws.com/{relative_path}"
    return config.public_url(relative_path)


_ALLOWED_ARTIFACT_RELEASE_TYPES = {"ci", "dev", "nightly", "prerelease"}

_ALLOWED_RELEASE_TYPES = {"dev", "nightly", "prerelease"}

_ALLOWED_RELEASE_BUCKET_TYPES = {"tarball", "python", "packages"}


def get_artifacts_bucket_config(
    release_type: str,
    repository: str,
    is_pr_from_fork: bool,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a repository.

    Args:
        release_type: "ci", "dev", "nightly", or "prerelease".
        repository: GitHub repository (e.g. "ROCm/TheRock").
        is_pr_from_fork: Whether this is a PR from a fork.

    Raises:
        ValueError: If release_type is invalid.
    """
    if release_type not in _ALLOWED_ARTIFACT_RELEASE_TYPES:
        raise ValueError(
            f"release_type={release_type!r} is invalid, "
            f"expected one of {_ALLOWED_ARTIFACT_RELEASE_TYPES}"
        )

    if release_type == "ci":
        if is_pr_from_fork or repository != "ROCm/TheRock":
            bucket_name = "therock-ci-artifacts-external"
        else:
            bucket_name = "therock-ci-artifacts"
    else:
        bucket_name = f"therock-{release_type}-artifacts"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def get_release_bucket_config(
    release_type: str,
    bucket_type: str,
) -> S3BucketConfig:
    """Look up the release bucket config for a given release type and bucket type.

    Args:
        release_type: "dev", "nightly", or "prerelease".
        bucket_type: "tarball", "python", or "packages".

    Returns:
        S3BucketConfig for the bucket ``therock-{release_type}-{bucket_type}``.

    Raises:
        ValueError: If release_type or bucket_type is invalid.
    """
    if release_type not in _ALLOWED_RELEASE_TYPES:
        raise ValueError(
            f"release_type={release_type!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_TYPES}"
        )
    if bucket_type not in _ALLOWED_RELEASE_BUCKET_TYPES:
        raise ValueError(
            f"bucket_type={bucket_type!r} is invalid, "
            f"expected one of {_ALLOWED_RELEASE_BUCKET_TYPES}"
        )
    bucket_name = f"therock-{release_type}-{bucket_type}"
    return _BUCKET_CONFIGS_BY_NAME[bucket_name]


def get_artifacts_bucket_config_for_workflow_run(
    github_repository: str,
    release_type: str | None = None,
    workflow_run_id: str | None = None,
    workflow_run: dict | None = None,
) -> S3BucketConfig:
    """Look up the artifacts bucket config for a workflow run.

    Combines environment-based inputs (RELEASE_TYPE, event payload) with
    optional workflow run metadata from the GitHub API to determine the
    correct artifacts bucket.

    Args:
        github_repository: GitHub repository (e.g. "ROCm/TheRock").
        release_type: Release type override. If None, reads RELEASE_TYPE
            from the environment (default: "ci").
        workflow_run_id: If set and ``workflow_run`` is None, fetches the
            workflow run from the GitHub API for fork detection.
        workflow_run: Optional workflow run dict from GitHub API. If
            provided, used directly for fork detection (no API call).
    """
    _log("Retrieving bucket info for workflow run...")
    _log(f"  github_repository: {github_repository}")

    if release_type is None:
        release_type = os.environ.get("RELEASE_TYPE", "ci")
    _log(f"  release_type: {release_type}")

    # Fetch workflow_run from API if not provided but workflow_run_id is set.
    # Deferred import: github_actions is an optional dependency not available in
    # all environments (e.g. local dev without the GHA support package installed).
    if workflow_run is None and workflow_run_id is not None:
        from github_actions.github_actions_api import (
            GitHubAPIError,
            gha_query_workflow_run_by_id,
        )

        try:
            workflow_run = gha_query_workflow_run_by_id(
                github_repository, workflow_run_id
            )
        except GitHubAPIError as e:
            run_url = (
                f"https://github.com/{github_repository}/actions/runs/{workflow_run_id}"
            )
            raise GitHubAPIError(
                f"Failed to query workflow run {workflow_run_id} in repository "
                f"{github_repository}: {run_url}\n"
                f"  {e}\n"
                f"Hint: Did you mean to specify a different repository with "
                f"--run-github-repo?"
            ) from e

    # Extract metadata from workflow_run if available
    if workflow_run is not None:
        _log(f"  workflow_run_id: {workflow_run['id']}")
        head_github_repository = workflow_run["head_repository"]["full_name"]
        is_pr_from_fork = head_github_repository != github_repository
        _log(f"  head_github_repository: {head_github_repository}")
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")
    else:
        # Deferred import: github_actions is optional in some environments;
        # only needed when resolving fork state from the on-disk event payload.
        from github_actions.github_actions_api import is_current_run_pr_from_fork

        is_pr_from_fork = is_current_run_pr_from_fork()
        _log(f"  is_pr_from_fork: {is_pr_from_fork}")

    config = get_artifacts_bucket_config(
        release_type=release_type,
        repository=github_repository,
        is_pr_from_fork=is_pr_from_fork,
    )
    _log(f"  bucket: {config.name}")

    # For fork PRs, skip OIDC and use runner base credentials instead.
    # Fork PRs cannot assume IAM roles via OIDC because they don't have
    # the required trust relationship. Return a config without an IAM role
    # so the configure-aws-credentials step is skipped.
    if is_pr_from_fork and config.iam_role is not None:
        _log("  Fork PR detected, skipping OIDC (using runner base credentials)")
        config = replace(config, iam_role=None)

    return config
