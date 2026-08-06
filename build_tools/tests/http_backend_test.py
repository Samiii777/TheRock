#!/usr/bin/env python
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for the read-only HTTPBackend."""

import functools
import hashlib
import http.server
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))

from _therock_utils import storage_location as storage_location_module
from _therock_utils.artifact_backend import HTTPBackend
from _therock_utils.workflow_outputs import WorkflowOutputRoot


class TestHTTPBackend(unittest.TestCase):
    def setUp(self):
        self.serve_dir = Path(tempfile.mkdtemp())
        self.payload = b"hello therock artifact"
        self.digest = hashlib.sha256(self.payload).hexdigest()

        self.root = WorkflowOutputRoot.for_local(run_id="local", platform="linux")
        self.art_key = "core-runtime_lib_generic.tar.xz"
        index_rel = self.root.artifact_index().relative_path
        art_rel = self.root.artifact(self.art_key).relative_path
        sha_rel = self.root.artifact(self.art_key + ".sha256sum").relative_path

        index_html = (
            f'<a href="{self.art_key}">{self.art_key}</a>'
            '<a href="other.txt">other.txt</a>'
        ).encode()
        for rel, content in [
            (index_rel, index_html),
            (art_rel, self.payload),
            (sha_rel, f"{self.digest}  {self.art_key}".encode()),
        ]:
            fp = self.serve_dir / rel
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_bytes(content)

        class QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *args, **kwargs):
                pass

        handler = functools.partial(QuietHandler, directory=str(self.serve_dir))
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.shutdown)

        base = f"http://127.0.0.1:{self.port}"
        saved = storage_location_module.StorageLocation.public_url
        self.addCleanup(
            setattr, storage_location_module.StorageLocation, "public_url", saved
        )
        storage_location_module.StorageLocation.public_url = property(
            lambda loc: f"{base}/{loc.relative_path}"
        )

        self.backend = HTTPBackend(output_root=self.root)

    def test_list_artifacts_parses_index(self):
        self.assertEqual(self.backend.list_artifacts(), [self.art_key])

    def test_list_artifacts_name_filter(self):
        self.assertEqual(self.backend.list_artifacts("core-runtime"), [self.art_key])
        self.assertEqual(self.backend.list_artifacts("nonexistent"), [])

    def test_download_verifies_sha256(self):
        dst = self.serve_dir / "out" / self.art_key
        self.backend.download_artifact(self.art_key, dst)
        self.assertEqual(dst.read_bytes(), self.payload)

    def test_download_rejects_sha256_mismatch(self):
        sha_rel = self.root.artifact(self.art_key + ".sha256sum").relative_path
        (self.serve_dir / sha_rel).write_bytes(
            b"0" * 64 + b"  " + self.art_key.encode()
        )
        with self.assertRaises(ValueError):
            self.backend.download_artifact(
                self.art_key, self.serve_dir / "out2" / self.art_key
            )

    def test_upload_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.backend.upload_artifact(Path("x"), "k")

    def test_copy_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            self.backend.copy_artifact("k", None)


if __name__ == "__main__":
    unittest.main()
