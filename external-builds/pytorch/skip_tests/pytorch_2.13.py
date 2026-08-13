# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Known failures on the PyTorch 2.13 wheels. These are already tracked in the
# other version skip files (pytorch_2.11.py - pytorch_2.12.py) and/or generic.py,
# but those exclusions are not picked up for the 2.13 version, so they are
# mirrored here. See
# https://github.com/ROCm/TheRock/issues/5596

skip_tests = {
    "common": {
        # Version skew: the pinned test_autograd.py leads the 2.13 wheel's
        # compiled core, exercising autograd behavior the wheel lacks. See
        # https://github.com/ROCm/TheRock/issues/26417
        "autograd": [
            # dict inputs to grad()/backward(): "all inputs have to be
            # Tensors or GradientEdges, but got str".
            "test_backward_dict_inputs",
            "test_backward_dict_inputs_tensor_backward",
            "test_grad_dict_inputs",
            "test_grad_dict_inputs_allow_unused",
            "test_grad_dict_inputs_batched_grads",
            "test_grad_dict_inputs_create_graph",
            "test_grad_dict_inputs_materialize_grads",
            "test_grad_dict_inputs_non_string_keys",
            "test_grad_dict_inputs_ordered_dict",
            # Function.apply kwargs: "apply() takes no keyword arguments".
            "test_custom_function_apply_kwargs",
            "test_custom_function_apply_kwargs_errors",
            "test_custom_function_apply_kwargs_required",
            "test_custom_function_apply_kwargs_setup_context",
            "test_custom_function_apply_kwargs_tensor",
            # empty-input validation ordering: ValueError vs expected RuntimeError.
            "test_grad_empty_inputs",
            "test_grad_dict_inputs_empty",
            # torch.autograd.enforce_grad_layout_policy semantics differ from wheel.
            "test_enforce_grad_layout_policy",
        ],
        "autograd": [
            # test/test_autograd.py from the pinned PyTorch checkout is newer
            # than the compiled core in the 2.13 wheel, so these exercise
            # autograd features/behavior that the wheel does not yet implement
            # (version skew, not a ROCm defect). Tracked in
            # https://github.com/ROCm/TheRock/issues/26417
            #
            # grad()/backward() with dict inputs - the wheel's compiled core
            # rejects non-Tensor inputs: "all inputs have to be Tensors or
            # GradientEdges, but got str".
            "test_backward_dict_inputs",
            "test_backward_dict_inputs_tensor_backward",
            "test_grad_dict_inputs",
            "test_grad_dict_inputs_allow_unused",
            "test_grad_dict_inputs_batched_grads",
            "test_grad_dict_inputs_create_graph",
            "test_grad_dict_inputs_empty",
            "test_grad_dict_inputs_materialize_grads",
            "test_grad_dict_inputs_non_string_keys",
            "test_grad_dict_inputs_ordered_dict",
            # autograd.Function.apply keyword arguments - the wheel's native
            # apply() raises "apply() takes no keyword arguments".
            "test_custom_function_apply_kwargs",
            "test_custom_function_apply_kwargs_errors",
            "test_custom_function_apply_kwargs_required",
            "test_custom_function_apply_kwargs_setup_context",
            "test_custom_function_apply_kwargs_tensor",
            # empty-inputs validation ordering differs: the wheel raises
            # ValueError where the test expects RuntimeError.
            "test_grad_empty_inputs",
        ],
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
