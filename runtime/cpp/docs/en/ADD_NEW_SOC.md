# ArcForge How to Add Support for a New SoC

This guide outlines the steps to add support for a new SoC (System on Chip) or cross-compilation target in the ArcForge build system.

**Method:** Create a toolchain file (using environment variables), update `CMakePresets.json`, and configure your environment.

---

## Step 1: Create a Toolchain File

Do not start from scratch; please copy an existing configuration.

1.  **Copy** the template:
    ```bash
    cp cmake/toolchains/rv1126bp.cmake cmake/toolchains/<new_soc_name>.cmake
    # Example: cp cmake/toolchains/rv1126bp.cmake cmake/toolchains/rk3588.cmake
    ```

2.  **Edit** the new file (`cmake/toolchains/rk3588.cmake`):
    *   **Modify Processor**: Update `CMAKE_SYSTEM_PROCESSOR` (e.g., `aarch64`, `arm`, `riscv64`).
    *   **Define SDK Environment Variable**: For example, if using Buildroot for RK SoC board SDK development, there will be a complete development environment (e.g., sysroot/toolchain, etc.) under `buildroot/output/rockchip_${new_soc_name}/host`.
        *   *Change from*: `if(DEFINED ENV{RV1126BP_SDK_ROOT})`
        *   *To*: `if(DEFINED ENV{RK3588_SDK_ROOT})`
    *   **Update Paths**: Adjust internal layout variables (such as `TOOLCHAIN_BIN_DIR`) to match your specific SDK structure.
    *   **Update Compilers**: Update the filenames for `gcc`, `g++`, `strip`, etc.

**Key Rule**: Never hardcode absolute paths (e.g., `/home/user/sdk`). Always rely on `$ENV{...}` variables or fallback default values.

---

## Step 2: Configure Your Environment

Since the toolchain file now relies on environment variables, you must define the location of the SDK on your specific machine.

1.  **Definition Method (Recommended)**:
    Modify the `.env` file in the project root directory (if this file does not exist, you need to go back to the first step of the entire project to create it. See [[README_ZH_CN.md# #### 1. 环境配置 (必做)]] for details).

    ```bash
    vim .env
    ```
    Add a new environment variable:
    ```bash
    ARC_${new_soc_name}_SDK_ROOT=".../xxxx"
    # Example:
    # ARC_RK3588S_SDK_ROOT="/development/docker_volumes/src/sdk/rk3588s-linux/buildroot/output/rockchip_rk3588/host"
    ```

---

## Step 3: Configure External Dependencies

### Dependency 1: Sherpa-Onnx
You must provide the download URL and SHA256 hash for the pre-compiled `sherpa-onnx` binaries specific to the new SoC.

1.  **Prepare Tarball**: (Assuming you have already compiled sherpa-onnx) Upload your pre-compiled `sherpa-onnx` tarball (containing `lib/` and `include/`) to a hosting location (e.g., GitHub Releases).
2.  **Get Hash**: Calculate the SHA256 hash of the file: `sha256sum filename.tar.xz`.
3.  **Create Dependency Mixin**: In `CMakePresets.json`, add a new entry under `configurePresets` to define these variables.

```json
    {
      "name": "mixin-deps-rk3588",
      "hidden": true,
      "description": "External dependencies URLs and Hashes for RK3588",
      "cacheVariables": {
        "SHERPA_ONNX_URL": "https://github.com/.../sherpa-onnx-v1.12.20-rk3588.tar.xz",
        "SHERPA_ONNX_HASH": "SHA256=a1b2c3d4e5..."
      }
    },
```

### Dependency 2: Librknnrt.so
1.  **Download Library File**: Download the file used by your new SoC from [RK's official repository](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api) (If the repository link is invalid, the old repository usually explicitly states the location of the new repository).
- **Note:**
  - Which version to download?
    - It depends on the version of the `librknnrt` library used when converting your sherpa-zipformer model. If the model uses a specific version of the rknn runtime library, you must remain consistent with it here.

2.  **Prepare Tarball**: Compress the `librknnrt.so` you just downloaded into a tarball and upload it to your own hosting location (e.g., GitHub Releases).
3.  **Get Hash**: Obtain or manually calculate the SHA256 hash of the file: `sha256sum filename.tar.xz`.
4.  **Create Dependency Mixin**: In `CMakePresets.json`, add a new entry under `configurePresets` to define these variables.

```json
    {
      "name": "mixin-deps-librknnrt-linux-aarch64",
      "hidden": true,
      "description": "External dependencies URLs and Hashes for librknnrt",
      "cacheVariables": {
        "LIBRKNNRT_URL": "https://github.com/../librknnrt-v2.2.0-linux-aarch64-shared-official.tar.xz",
        "LIBRKNNRT_HASH": "SHA256=d933df1daeb112d6c..."
      }
    },
```

---

## Step 4: Register in CMakePresets.json

Now define the platform configuration and the final build target, inheriting the dependency Mixin you created above.

Open `CMakePresets.json` and add entries to the `configurePresets` array.

### 4.1. Add "Platform" Preset (Hidden)
This links the logical name to the toolchain file.

```json
    {
      "name": "plat-rk3588",
      "hidden": true,
      "description": "Rockchip RK3588 Target",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/cmake/toolchains/rk3588.cmake"
      }
    },
```

### 4.2. Add Concrete Build Presets
Define the actual build targets.
**Important**: You must inherit the dependency Mixin (`mixin-deps-rk3588`) here.

```json
    {
      "name": "rk3588-release",
      "inherits": ["base", "plat-rk3588", "mixin-deps-rk3588", "cfg-release", "cfg-shared", "mixin-deps-librknnrt-linux-aarch64"],
      "installDir": "${sourceDir}/install/rk3588/release",
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "${sourceDir}/install/rk3588/release"
      }
    },
    {
      "name": "rk3588-debug",
      "inherits": ["base", "plat-rk3588", "mixin-deps-rk3588", "cfg-debug", "cfg-shared", "mixin-deps-librknnrt-linux-aarch64"],
      "installDir": "${sourceDir}/install/rk3588/debug",
      "cacheVariables": {
        "CMAKE_PREFIX_PATH": "${sourceDir}/install/rk3588/debug"
      }
    }
```

---

## Step 5: Verify and Build

1.  **Check Visibility**:
    Run the interactive build script or list presets to see if your new SoC appears.
    ```bash
    ./build.sh list
    ```
    *You should see `rk3588-release` and `rk3588-debug` in the list.*

2.  **Run Build**:
    ```bash
    ./build.sh cb rk3588-release
    ```

---

## Summary Checklist

- [ ] **Toolchain**: Created `cmake/toolchains/xxx.cmake` (copied from template).
- [ ] **Env Vars**: Updated the specific `ENV{XXX_SDK_ROOT}` in the toolchain file and exported it in the Shell.
- [ ] **Dependencies**: Added `mixin-deps-xxx` in `CMakePresets.json` with valid `SHERPA_ONNX_URL` and `HASH`.
- [ ] **Presets**: Added `plat-xxx` and `xxx-release` (inheriting `mixin-deps-xxx`).
- [ ] **Sync**: Ensured `installDir` in presets matches `CMAKE_PREFIX_PATH`.