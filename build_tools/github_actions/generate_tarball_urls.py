#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate presigned download URLs for uploaded tarballs.

This resolves the uploaded tarball destination using WorkflowOutputRoot,
verifies the expected tarball objects exist in S3, and writes presigned
download URLs to GitHub Actions outputs.

Outputs written to GITHUB_OUTPUT:
    tarball_url
    jax_tarball_url
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3

_BUILD_TOOLS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BUILD_TOOLS_DIR))

from _therock_utils.workflow_outputs import WorkflowOutputRoot
from github_actions_api import gha_set_output


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate presigned download URLs for uploaded tarballs"
    )
    parser.add_argument("--run-id", required=True, help="Workflow run ID")
    parser.add_argument(
        "--platform",
        required=True,
        choices=["linux", "windows"],
        help="Platform for workflow outputs",
    )
    parser.add_argument(
        "--release-type",
        default="",
        help='Release type: "" for CI, or "dev", "nightly", "prerelease"',
    )
    parser.add_argument(
        "--package-version",
        required=True,
        help="ROCm/TheRock package version used in tarball names",
    )
    parser.add_argument(
        "--dist-amdgpu-families",
        required=True,
        help="Semicolon-separated family list used for tarball generation",
    )
    parser.add_argument(
        "--jax-amdgpu-family",
        default="",
        help="Selected JAX AMDGPU family for family-specific tarball URL",
    )
    parser.add_argument(
        "--aws-region",
        default="us-east-2",
        help="AWS region for S3 client",
    )
    parser.add_argument(
        "--expires-in",
        type=int,
        default=7 * 24 * 60 * 60,
        help="Presigned URL lifetime in seconds",
    )
    return parser.parse_args(argv)


def parse_family_list(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(";") if name.strip()]


def get_tarball_names(
    *,
    platform: str,
    package_version: str,
    families: list[str],
    jax_family: str,
) -> tuple[str, str]:
    if len(families) == 1:
        tarball_name = f"therock-dist-{platform}-{families[0]}-{package_version}.tar.gz"
    else:
        tarball_name = f"therock-dist-{platform}-multiarch-{package_version}.tar.gz"

    jax_tarball_name = ""
    if jax_family:
        jax_tarball_name = (
            f"therock-dist-{platform}-{jax_family}-{package_version}.tar.gz"
        )

    return tarball_name, jax_tarball_name


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Unexpected S3 URI: {s3_uri}")
    bucket_and_key = s3_uri[len("s3://") :]
    bucket, prefix = bucket_and_key.split("/", 1)
    return bucket, prefix


def generate_presigned_url(
    *,
    s3_client,
    bucket: str,
    key: str,
    expires_in: int,
) -> str:
    # Fail fast if the uploaded object is not present where we expect it.
    s3_client.head_object(Bucket=bucket, Key=key)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    families = parse_family_list(args.dist_amdgpu_families)

    output_root = WorkflowOutputRoot.from_workflow_run(
        run_id=args.run_id,
        platform=args.platform,
        release_type=args.release_type or None,
    )
    dest = output_root.tarballs()
    bucket, prefix = parse_s3_uri(dest.s3_uri)

    tarball_name, jax_tarball_name = get_tarball_names(
        platform=args.platform,
        package_version=args.package_version,
        families=families,
        jax_family=args.jax_amdgpu_family.strip(),
    )

    s3 = boto3.client("s3", region_name=args.aws_region)

    tarball_key = f"{prefix}/{tarball_name}"
    tarball_url = generate_presigned_url(
        s3_client=s3,
        bucket=bucket,
        key=tarball_key,
        expires_in=args.expires_in,
    )

    jax_tarball_url = ""
    if jax_tarball_name:
        jax_tarball_key = f"{prefix}/{jax_tarball_name}"
        jax_tarball_url = generate_presigned_url(
            s3_client=s3,
            bucket=bucket,
            key=jax_tarball_key,
            expires_in=args.expires_in,
        )

    gha_set_output(
        {
            "tarball_url": tarball_url,
            "jax_tarball_url": jax_tarball_url,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
