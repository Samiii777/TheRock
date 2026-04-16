# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

if(THEROCK_ENABLE_HIPFILE)
  ##############################################################################
  # hipFile
  ##############################################################################
  set(_hipfile_subproject_names)
  therock_cmake_subproject_declare(hipfile
    USE_TEST_AMDGPU_TARGETS
    EXTERNAL_SOURCE_DIR "${THEROCK_ROCM_SYSTEMS_SOURCE_DIR}/projects/hipfile"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/hipfile"
    # High latency LTO link of a single library.
    BACKGROUND_BUILD
    CMAKE_ARGS
      -DCMAKE_VERBOSE_MAKEFILE=OFF
      -DUSE_RO=ON
      -DUSE_IPC=ON
      -DUSE_GDA=ON
      -DGDA_MLX5=ON
      -DGDA_BNXT=ON
      -DGDA_IONIC=ON
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      "-DEXPLICIT_ROCM_VERSION=${ROCM_MAJOR_VERSION}.${ROCM_MINOR_VERSION}.${ROCM_PATCH_VERSION}"
      -DBUILD_FUNCTIONAL_TESTS=${THEROCK_BUILD_TESTING}
      -DBUILD_UNIT_TESTS=${THEROCK_BUILD_TESTING}
      -DSOURCE_DIR=${_source_dir}
      -DPATCHELF=${PATCHELF}
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
    CMAKE_INCLUDES
      therock_explicit_finders.cmake
    COMPILER_TOOLCHAIN
      amd-hip
    BUILD_DEPS
      rocm-cmake
      therock-fmt
      therock-googletest
      ${THEROCK_BUNDLED_NUMACTL}
    RUNTIME_DEPS
      hip-clr
  )
  therock_cmake_subproject_glob_c_sources(hipfile
    SUBDIRS
      src
  )
  therock_cmake_subproject_provide_package(hipfile HIPFILE lib/cmake/hipfile)
  therock_cmake_subproject_activate(hipfile)
  list(APPEND _hipfile_subproject_names hipfile)

  therock_provide_artifact(hipfile
    TARGET_NEUTRAL
    DESCRIPTOR artifact-hipfile.toml
    COMPONENTS
      dbg
      dev
      doc
      lib
      test
    SUBPROJECT_DEPS
      ${_hipfile_subproject_names}
  )
endif(THEROCK_ENABLE_HIPFILE)
