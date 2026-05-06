# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES", "")

logging.basicConfig(level=logging.INFO)

# Issue #5047: gfx1151 segfaults under high parallelism (UMA APU memory pressure
# plus repeated queue create/destroy hitting ROCR-Runtime queue scratch race).
ctest_parallelism = "8"
if AMDGPU_FAMILIES == "gfx1151":
    ctest_parallelism = "1"

cmd = [
    "ctest",
    "--test-dir",
    f"{THEROCK_BIN_DIR}/hipRAND",
    "--output-on-failure",
    "--parallel",
    ctest_parallelism,
]
logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")

subprocess.run(
    cmd,
    cwd=THEROCK_DIR,
    check=True,
)
