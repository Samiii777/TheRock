# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch 2.13 wheels. These are already tracked in the
# other version skip files (pytorch_2.11.py - pytorch_2.12.py) and/or generic.py,
# but those exclusions are not picked up for the 2.13 version, so they are
# mirrored here. See
# https://github.com/ROCm/TheRock/issues/5596

skip_tests = {
    "common": {
        "cuda": [
            # TestCuda - conflicts with how our test script and runners are
            # configured.
            "test_hip_device_count",
            # TestCudaAllocator - passes on single run, crashes if run in a
            # group. TypeError: 'CustomDecompTable' object is not a mapping
            "test_memory_compile_regions",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator'. The hipblas.h include error persists in the
            # ROCm SDK environment:
            #   fatal error: 'hipblas/hipblas.h' file not found
            "test_mempool_empty_cache_inactive",
            # TestMemPool - RuntimeError: Error building extension
            # 'dummy_allocator_v1' (same hipblas.h include error)
            "test_mempool_limited_memory_with_allocator",
        ],
        "ops": [
            # TestCommonCUDA - upstream test bug, not a HIP kernel defect.
            # test_compare_cpu applies a single hardcoded atol=rtol=1e-3 to every
            # op with no per-op override. Measured against a float64 reference,
            # bicubic grid_sample has ~0.011 of inherent fp32 error (10x that
            # tolerance) and the GPU result is closer to ground truth than the CPU
            # one on the majority of elements, so a stray element exceeding 1e-3
            # is fp32 FMA/ordering noise rather than a wrong result.
            #   AssertionError: Tensor-likes are not close!
            #   Mismatched elements: 1 / 943920 (0.0%)
            #   Greatest absolute difference: 0.002597808837890625 (up to 0.001 allowed)
            # Remove once test_compare_cpu grows a per-op tolerance for
            # nn.functional.grid_sample upstream.
            "test_compare_cpu_nn_functional_grid_sample_cuda_float32",
            # TestFakeTensorCUDA - upstream FakeTensor bug, unrelated to ROCm.
            # Both the amp and no_amp variants fail identically, so autocast
            # is incidental to the failure.
            # Reproduces identically on a stock CPU-only torch wheel with no AMD
            # GPU present. Triggered by the zero-element OpInfo sample
            # ((0, 8) x (0, 8)): the fake aten.view.default of _trilinear's
            # zero-element output gets a fresh 0-byte storage instead of aliasing
            # its input, because zero-byte storages all share data_ptr 0 and so
            # cannot be correlated by MetaConverter.storage_memo. Metadata-only;
            # a 0-byte storage holds no data, so nothing is numerically wrong.
            #   torch._subclasses.fake_tensor.MetadataMismatchError: When comparing
            #   the output of aten.view.default on FakeTensor and concrete Tensors,
            #   found mismatch in outputs_alias_inputs check False != True
            "test_fake_crossref_backward_amp_nn_functional_bilinear_cuda_float32",
            "test_fake_crossref_backward_no_amp_nn_functional_bilinear_cuda_float32",
        ],
        "nn": [
            # TestNNDeviceTypeCUDA - AssertionError: Scalars are not close!
            # Expected 3.875156879425049 but got 3.876049757003784.
            # Absolute difference: 0.0008928775787353516 (up to 1e-05 allowed)
            # Relative difference: 0.0002304106921389532 (up to 1.3e-06 allowed)
            "test_CTCLoss_cudnn_cuda",
            # TestNNDeviceTypeCUDA - cudnn CTC loss numerical mismatch
            "test_ctc_loss_cudnn_tensor_cuda_cuda",
            # TestNNDeviceTypeCUDA - per-call dropout randomness mismatch
            "test_LSTM_dropout_per_call_randomness_dropout_p_0_5_training_True_cuda",
            # TestNNDeviceTypeCUDA - upsampling launch failure on gfx950
            # Separately tracked in https://github.com/ROCm/TheRock/issues/5270
            "test_upsamplingNearest2d_launch_rocm_cuda",
        ],
    },
}
