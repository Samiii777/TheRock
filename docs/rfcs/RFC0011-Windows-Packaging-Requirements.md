---
author: Liam Berry (LiamfBerry), Saad Rahim (saadrahim)
created: 2026-03-13
modified: 2026-03-13
status: draft
---

# TheRock Windows Packaging Requirements

With the implementation of TheRock build system, native Windows packaging and installation requirements must be defined to complement the Linux software packaging requirements. This RFC defines the packaging, installation, upgrade, uninstallation, versioning, and distrobuition requirements for TheRock software on native Windows, including the ROCm Core SDK and related ROCm software components.

Our goals are to:

1. **Standardize packaging behaviour for native Windows ROCm sotware**
1. **Ensure predictable installation, upgrade, repair, side-by-side support, and uninstall behavior**
1. **Provide redistributable-friendly Windows delivery mechanisms for developers, IT administrators, and ISVs**
1. **Align Windows packaging structure with the broader TheRock cross-platform packaging model where practical**
1. **Support automated artifact generation and productized deliverables from TheRock**

## Scope

### In Scope

- Native Windows packaging requirements from ROCm software build with TheRock
- MSI-based package requirements
- Winget package and meta-package requirements
- Python pip package requirements for Windows developer workflows
- Portable ZIP package requirements
- Runtime, development, and developer-tools package separation
- Installation directory layout and package granularity
- Side-by-side installation policy for major.minor versions
- Patch upgrade behavior
- Environment variables, registry keys, and discovery mechanisms
- Logging, signing, and Windows-specific installation semantics
- Guidance for legacy System32 runtime cleanup and migration

### Out of Scope

- Windows display driver packaging and WHQL process details
- WSL and WSL2 package requirements
- Internal CI/CD implementation details
- Full feature parity planning for every ROCm component on Day 1
- Legacy HIP SDK packaging behavior except where migration handling is required
- Microsoft Store-specific package requirements

## Windows Packaging Requirements

### Packaging Formats

Windows ROCm software must be delivered using packaging formats appropriate to the Windows ecosystem while preserving the same general package boundaries expected from TheRock on Linux.

The supported packaging formats are:

- **MSI packages** as the primary OS-integrated installation unit
- **Winget packages** as Windows package-manger-facing meta packages or installers that reference AMD-hosted MSI artifacts
- **Python pip packages** for Python bindings, Python-first tooling, and environment-scoped developer workflows
- **ZIP archives** for portable, offline, CI, ir power-user scenarios

MSI packages are the authoritative Windows installation unit. Winget, pip, and ZIP deliverables must complement MSI behavior rather than redefine the core packaging contract.

### Directory Layout

The ROCm Core SDK on Windows must be installed under a versioned installation root to support side-by-side installation of major.minor releases.

```
C:\rocm\core-X.Y
```

Where:

- `X.Y` is the major and minor version
- Patch versions must be installed in place within the existing `X.Y` directory
- Side-by-side installation is supported for different major.minor versions
- Patch-only side-by-side installation is not supported

The installed directory structure must mirror the cross-platform ROCm layout as closely as practical:

```
C:\rocm\core-X.Y\
  bin\
  lib\
  include\
  share\
  tools\
  version.txt
```

A convenience path to the most recently installed version should be maintained when practical:

```
C:\rocm\core  ->  C:\rocm\core-8.2
C:\rocm\core-8  ->  C:\rocm\core-8.2
C:\rocm\raytracing-8  ->  C:\rocm\raytracing-8.2
```

This allows users, scripts, and build systems to either target and latest installed release or pin to a major line while still preserving independently versioned install roots.

### Decouple User Space from Adrenaline Driver

Windows packages must avoid reliance on `C:\Windows\System32` for ROCm runtime discovery.

All new Windows ROCm runtime components must be installed into the package installation root, primarily under `bin`, and discovered through one or more of the following supported mechanisms:

- Application-local deployment for redistributable scenarios
- `PATH` entries associated with the selected ROCm installation
- Registry-based SDK discovery
- Environment-variable-based SDK discovery

New installations must not place core ROCm runtime DLLs into `System32`. Legacy driver-installed runtime DLLs in `System32` that conflict with the new Windows packaging model must be detected and handled by the appropriate runtime installer. At a minimum, the Windows runtime package must handle cleanup of legacy `amdhip64` and `amd_comgr` placements when present, while preserving installer robustness if files are locked or permissions are insufficient.

### Package Naming

Windows package naming should remain aligned with the Linux TheRock naming model where practical so that users can reason about package purpose consistently across operating systems.

The `amdrocm-` naming prefix should be used for AMD-published Windows package componens where a package-level identity is exposed directly to users.

Examples inlcude:

- `amdrocm-runtimes`
- `amdrocm-core`
- `amdrocm-core-dev`
- `amdrocm-developer-tools`
- `amdrocm-core-sdk`
- `amdrocm-raytracing`

Winget package identifiers may use Windows ecosystem naming conventions such as `AMD.ROCm`, but they should map cleanly to the same product and component boundaries.

### Package Granularity

Windows package granularity should follow the same general model as Linux: runtime and development responsibilities must be seperable, and developer tools must be independently installable.

The following high-level package groupings must be avaialable:

| Name                      | Content                                                                                            | Description |
| :------------------------ | :------------------------------------------------------------------------------------------------- | :------------- |
| `amdrocm-runtimes`        | HIP runtime, runtime compiler support, required runtime libraries                                  |                |
| `amdrocm-core`            | Runtime components, core libraries, core utilities, discovery tools                                |                |
| `amdrocm-core-dev`        | Headers, CMake config files, import libraries, static libraries, compiler-facing development files |                |
| `amdrocm-developer-tools` | Debugging, profiling, diagnostics, and related developer tools                                     |                |
| `amdrocm-core-sdk`        | Core runtime, development files, and developer tools                                               |                |

Windows package composition may evolve as TheRock matures, but the runtime vs. development vs. tools split must remain clear.

### Installation Configurations

Windows installation flows must support multiple install configurations aligned with Windows user expectations and the ROCm SDK for Windows product direction.

At minimum, the following install configurations must be supported:

- **Core SDK**: default developer installation
- **Runtime Only**: minimal runtime footproint for executing prebuilt applications and redistribuition workflows
- **Developer Tools Only**: debugging, profiling, and diagnostics when runtime is already present
- **Custom**: component-level selection for advanced users where supported by package dependency rules

These configurations may be implemented as separate packages, features, meta packages, or a combination thereof, but their behavior must remain well-defined and documented.

### MSI Package Requirements

MSI packages are the primary Windows installation mechanism and must satisfy the following requirements:

- Support silent installation
- Support logging
- Support default installation path behavior
- Support custom installation path overrides
- Support uninstall and repair flows through standard Windows Installer semantics
- Support per-machine installation by default
- Support per-user installation where technically valid for the selected package set
- Be digitally signed by AMD
- Avoid partially installed states and maintain transactional integrity to the degree supported by Windows Installer

Example installation commands:

```
msiexec /i amdrocm-core-sdk.msi /quiet
msiexec /i amdrocm-core-sdk.msi INSTALLDIR="D:\tools\rocm\core-8.2" /quiet
msiexec /x amdrocm-core-sdk.msi /quiet
msiexec /famus amdrocm-core-sdk.msi /quiet
```

MSI installers should be GUI-less or minimal-UI by default and must support unattended enterprise deployment.

### Installation Logic and Version Handling

Upon execution, MSI packages must inspect the installation target and apply deterministic version-handling rules.

The following behavior matrix must be supported:

| Scenario                                                 | Behavior                                                                                           |
| :------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| No ROCm installation at target path                      | Installs normally                                                                                  |
| Older version detected at the same target path           | Perform in-place upgrade                                                                         |
| Same version detected at the same target path            | Return success with no action, unless an explicit repair or reinstall mode is requested |
| Newer version detected at the same target path           | Abort with error and instruct the user to uninstall or choose a different path                   |
Different major.minor version detected at a different path | Allow side-by-side installation                                                                    |

Path versions must upgrade in place within the same `X.Y` installation root.

Major.minor releases must be installable side by side in distinct versioned roots.

### Upgrade and Uninstall Requirements

Windows packages must provide predictable and clean upgrade and uninstall behavior.

Upgrade requirements:

- In-place upgrades must preserve the expected installation root for the target `X.Y` line
- Upgrade flows must avoid overwriting unrelated user environment settings
- Upgrade flows must not leave stale package-owned files in shared locations when the package manager can safely remove them
- Package upgrade behavior must remain consistent whether invoked directly through MSI or through winget

Uninstall requirements:

- Remove files owned by the installation being removed
- Remove environment-variable updates owned by that installation if they still reference that installation
- Remove registry entries created by that installation
- Remove package-owend  `PATH` entries associated with that installation only
- Avoid impacting other installed ROCm major.minor versions

### Environment Variables

After successful installation, Windows installers must publish a stable discovery mechanism for tools and build systems.

At minimum:

- `ROCM_PATH` must point to the installation root of the most recently installed active ROCm version
- The selected installation's `bin` directory must be prepended to the relevant `PATH`
- Duplicate `PATH` entries must not be introduced across reinstalls or upgrades
- Per-machine installs must modify machine-scoped enrionment variables
- Per-user installs must modify user-scoped environment variables only

The convenience variable `ROCM_PATH` is last-writer-wins. Build systems and applications that require deterministic selection of a specific version should rely on versioned install paths and registry-based discovery rather than assuming `ROCM_PATH` is pinned permanently.

### Registry Requirements

To support discovery and side-by-side versioning, Windows ROCm installers must create versioned registry keys.

Per-machine installs:

```
HKLM\Software\AMD\ROCm\X.Y\
```

Per-user installs:

```
HKCU\Software\AMD\ROCm\X.Y\
```

At minimum, each versioned key must contain:

- `InstallDir`
- `ProductCode`
- `Version`

Additionally, a convenience pointer to the latest installed active version should be maintained:

```
HKLM\Software\AMD\ROCm\CurrentVersion
HKCU\Software\AMD\ROCm\CurrentVersion
```

This convenience pointer is also last-writer-wins and exists to support straightforward SDK dsicovery by tools and administrators.

### Driver Compatibility and Preflight

Driver packaging is out of scope for this RFC, but Windows SDK packaging must be designed around an explicit driver compatibility contract.

Windows ROCm packages must:

- Publish a compatibility matrix describing supported driver ranges for each ROCm release line
- Avoid coupling SDK patch delivery to mandatory driver rebundling wherever possible
- Provide install-time or first-run preflight checks that warn when the installed driver is outside the supported compatiblity range

The Windows packaging contract must assume that the display driver and the ROCm SDK are seperate deliverables, even where an AMD driver may bundle or involve installation of a runtime-oriented package or compatibility purposes. It should be noted that users are expected to self install the driver in accordance with this.

### Winget Requirements

Winget packages are the Windows package-manager-facing distribution layer for ROCm.

Winget packages must:

- Reference AMD-hosted MSI installers
- Include version metadata and installer hash validation
- Support silent installation flows
- Preserve MSI-defined upgrade and uninstall behavior
- Support installation-path override capability where the underlying MSI supports it
- Maintain clear package naming and publisher metadata

A top-level package identifier such as `AMD.ROCm` may be used for the primary Windows SDK experience. Additional componentized identifiers may be introduced if needed, but must remain aligned with the same package boundaries defined by this RFC.

### Visual Studio Code Plugin Requirements

A Visual Studio plugin must for ROCm must support deterministic discovery of the ROCm toolchain and assiociated build binaries on Windows. The plugin must support two binding modes:

**Bind built binaries with latest**
The plugin resolves the SDK/toolchain root from an environment variable, in this case `ROCM_PATH`. This allows projects to automatically build against the most recently installed ROCm version.

**Bind fixed version using registry keys**
The plugin resolves the SDK/toolchain root from a version-specific installation record (e.g., Windows registry or installer metadata). This allows projects to bind to a specific ROCm version for reproducible builds and CI environments.

The plugin must clearly indicate the resolved SDK path and version used for the build.

### Python Pip Requirements

Python packages serve Python-first developer workflows and environment-scoped distribution scenarios.
Windows pip packages must satisfy the following requirements:

- Be environment-scoped and must not modify system-wide registry keys
- Must not modify system or user `PATH` outside the active Python environment
- Must not set or mutate `ROCM_PATH`
- Use consistent naming aligned with ROCm release versioning
- Support standard Windows Python packaging semantics including virtual environments
- Support offline installation from mirrored package sources

Pip packages may include:

- Python bindings
- Python developer tooling
- Console entry points installed into the Python environment
- Narrowly scoped runtime components needed for Python-first workflows

Heavy native runtime delivery and general-purpose Windows SDK installation must remain centered on MSI and ZIP packaging.

### ZIP Package Requirements

ZIP packages must provide a portable file-tree representation of a Windows ROCm installation.

ZIP archive layout must match the labelled directory layout:

```
rocm-core-X.Y.Z.zip
  rocm-core-X.Y\bin\...
```

ZIP packages:

- Must not modify environment variables
- Must not modify `PATH`
- Must not create registry entries
- Must remain suitable for CI, offline deployment, and advanced users

Tools and scripts inside ZIP packages should function correctly when the extracted directory is used directly as an SDK root.

### Logging Requirements

Windows installers must provide detailed logging for installation, upgrade, repair, uninstall, and cleanup actions.

Installer logs must include, at minimum:

- Installation path selection
- Exisiting-version detection
- Version comparison result
- Environment-variable updates
- Registry writes and removals
- Legacy runtime cleanup actions when applicable
- Reboot scheduling if cleanup of locked files requires deffered removal

Example:

```
msiexec /i amdrocm-core-sdk.msi /l*vx install.log
```

### Security and Signing Requirements

All Windows ROCm distribution artifacts must follow AMD signing and transport requirements.

- MSI installers must be digitally signed by AMD
- AMD-hosted package endpoints must use HTTPS
- Winget manifests must include hash validation
- Python package and ZIP archives should be published with metadata, reproducibility, and integrity verification

### Redistribution Requirements

Windows packaging must support a clear redistribution story for ISVs without requiring every end user to install a full development SDK.
The supported redistrobution models are:

1. **Application-local bundled runtime files** for supported runtime subsets
1. **Environment-scoped pip packages** for Python-first workflows
1. **Optional runtime-orianted MSI or winget install path** for customers who prefer a system-installed runtime

Redistrobution documentation must clearly distringuish:

- Development SDK installation
- Runtime-only installation
- Application-local redistrobution
- Python environment-scoped installation
