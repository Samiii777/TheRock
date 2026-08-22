#!/bin/bash

# run_fused_layer_norm exports through torch.onnx; without these the export
# tests cannot run.
python -m pip install --quiet "onnx>=1.16" onnxscript

APEX_TEST_WITH_ROCM=1 APEX_SKIP_FLAKY_TEST=1 python run_test.py
