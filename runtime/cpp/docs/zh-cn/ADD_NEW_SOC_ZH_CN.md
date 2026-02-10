# ArcForge 如何添加对新 SoC 的支持

本指南概述了在 ArcForge 构建系统中添加对新 SoC（片上系统）或交叉编译目标支持的步骤。

**方法：** 创建工具链文件（使用环境变量），更新 `CMakePresets.json`，并配置你的环境。

---

## 步骤 1：创建工具链文件

不要从零开始；请复制现有的配置。

1.  **复制**模板：
    ```bash
    cp cmake/toolchains/rv1126bp.cmake cmake/toolchains/<new_soc_name>.cmake
    # 示例: cp cmake/toolchains/rv1126bp.cmake cmake/toolchains/rk3588.cmake
    ```

2.  **编辑**新文件 (`cmake/toolchains/rk3588.cmake`)：
    *   **修改处理器 (Processor)**：更新 `CMAKE_SYSTEM_PROCESSOR`（例如：`aarch64`, `arm`, `riscv64`）。
    *   **定义SDK环境变量**：例如RK的SOC用Buildroot进行板子SDK开发的话，那么在buildroot/output/rockchip_${new_soc_name}/host下会有完整的开发环境（如sysroot/toolchain等）
        *   *将*：`if(DEFINED ENV{RV1126BP_SDK_ROOT})`
        *   *改为*：`if(DEFINED ENV{RK3588_SDK_ROOT})`
    *   **更新路径**：调整内部布局变量（如 `TOOLCHAIN_BIN_DIR`）以匹配你特定的 SDK 结构。
    *   **更新编译器**：更新 `gcc`, `g++`, `strip` 等的文件名。

**关键规则**：永远不要硬编码绝对路径（如 `/home/user/sdk`）。始终依赖 `$ENV{...}` 变量或回退默认值。

---

## 步骤 2：配置你的环境

由于工具链文件现在依赖于环境变量，你必须定义 SDK 在你特定机器上的位置。

1.  **定义方式（推荐）**：
    修改项目根目录的.env（若无此文件，需要回到整个项目的第一步进行创建。详细查看[[README_ZH_CN.md# #### 1. 环境配置 (必做)]]

    ```bash
    vim .env
    ```
    新增一个环境变量：
    ```bash
    ARC_${new_soc_name}_SDK_ROOT=".../xxxx"
    # 例如:
    # ARC_RK3588S_SDK_ROOT="/development/docker_volumes/src/sdk/rk3588s-linux/buildroot/output/rockchip_rk3588/host"
    ```

---

## 步骤 3：配置外部依赖

### 依赖1：Sherpa-Onnx
你必须为特定于新 SoC 的预编译 `sherpa-onnx` 二进制文件提供下载 URL 和 SHA256 哈希值。

1.  **准备 Tar 包**：(此处假设你已经编译完sherpa-onnx)将你的预编译 `sherpa-onnx` tar 包（包含 `lib/` 和 `include/`）上传到托管位置（例如 GitHub Releases）。
2.  **获取哈希值**：计算文件的 SHA256 哈希值：`sha256sum filename.tar.xz`。
3.  **创建依赖 Mixin**：在 `CMakePresets.json` 中，在 `configurePresets` 下添加一个新条目来定义这些变量。

```json
    {
      "name": "mixin-deps-rk3588",
      "hidden": true,
      "description": "RK3588 的外部依赖 URL 和哈希值",
      "cacheVariables": {
        "SHERPA_ONNX_URL": "https://github.com/.../sherpa-onnx-v1.12.20-rk3588.tar.xz",
        "SHERPA_ONNX_HASH": "SHA256=a1b2c3d4e5..."
      }
    },
```

### 依赖2：Librknnrt.so
1.  **下载库文件**：[从RK的官方仓库](https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api)（仓库链接若失效，一般在旧仓库会明确告知新仓库的位置）下载到你新的SOC所使用的文件。
- **注意点：**
  - 下载哪一个版本？
    - 要看你使用的sherpa-zipformer模型转换时所使用的librknnrt库的版本。模型rknn使用了哪一个版本的runtime库，你此处就得跟它保持一致。

2.  **准备 Tar 包**：将你刚才下载的 `librknnrt.so` 压缩成 tar 包上传到你自己的托管位置（例如 GitHub Releases）。
2.  **获取哈希值**：获取或自行计算文件的 SHA256 哈希值：`sha256sum filename.tar.xz`。
3.  **创建依赖 Mixin**：在 `CMakePresets.json` 中，在 `configurePresets` 下添加一个新条目来定义这些变量。

```json
    {
      "name": "mixin-deps-librknnrt-linux-aarch64",
      "hidden": true,
      "description": "External dependencies URLs and Hashes for librknnrt ",
      "cacheVariables": {
        "LIBRKNNRT_URL": "https://github.com/../librknnrt-v2.2.0-linux-aarch64-shared-official.tar.xz",
        "LIBRKNNRT_HASH": "SHA256=d933df1daeb112d6c..."
      }
    },
```

---

## 步骤 4：在 CMakePresets.json 中注册

现在定义平台配置和最终构建目标，并继承你上面创建的依赖 Mixin。

打开 `CMakePresets.json` 并向 `configurePresets` 数组添加条目。

### 4.1. 添加 “平台 (Platform)” 预设（隐藏）
这将逻辑名称链接到工具链文件。

```json
    {
      "name": "plat-rk3588",
      "hidden": true,
      "description": "Rockchip RK3588 目标",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "${sourceDir}/cmake/toolchains/rk3588.cmake"
      }
    },
```

### 4.2. 添加具体构建预设
定义实际的构建目标。
**重要**：你必须在此处继承依赖 Mixin (`mixin-deps-rk3588`)。

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

## 步骤 5：验证并构建

1.  **检查可见性**：
    运行交互式构建脚本或列出预设，查看你的新 SoC 是否出现。
    ```bash
    ./build.sh list
    ```
    *你应该会在列表中看到 `rk3588-release` 和 `rk3588-debug`。*

2.  **运行构建**：
    ```bash
    ./build.sh cb rk3588-release
    ```

---

## 总结清单

- [ ] **工具链**：已创建 `cmake/toolchains/xxx.cmake`（从模板复制）。
- [ ] **环境变量**：已更新工具链文件中的特定 `ENV{XXX_SDK_ROOT}` 并在 Shell 中导出。
- [ ] **依赖**：已在 `CMakePresets.json` 中添加 `mixin-deps-xxx`，并包含有效的 `SHERPA_ONNX_URL` 和 `HASH`。
- [ ] **预设**：已添加 `plat-xxx` 和 `xxx-release`（继承 `mixin-deps-xxx`）。
- [ ] **同步**：已确保预设中的 `installDir` 与 `CMAKE_PREFIX_PATH` 匹配。
