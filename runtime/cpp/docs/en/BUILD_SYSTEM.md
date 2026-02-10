# ArcForge Build System Explained

This document aims to reveal the build system architecture of the ArcForge project. The project adopts the **"Modern CMake"** philosophy, achieving automation and standardization of build logic through a highly encapsulated DSL (Domain-Specific Language).

---

## 1. Core Architecture View

### 1.1 Module Dependency Topology (Module Topology)
ArcForge employs a strict layered architecture. Lower-level modules are invisible to upper layers, while upper layers inherit lower-level functionalities through interfaces.

<!-- Please paste the Mermaid code for [Figure 3: Module Dependency Topology (Dependency Graph)] here -->
<!-- Suggest using graph BT or LR layout -->
```mermaid
graph BT
    %% ==========================================
    %% Styles Definition (Figure 3 Specific)
    %% ==========================================
    classDef app fill:#0d47a1,stroke:#90caf9,stroke-width:2px,color:#fff;
    classDef lib fill:#e65100,stroke:#ffcc80,stroke-width:2px,color:#fff;
    classDef third fill:#333333,stroke:#ffffff,stroke-width:2px,color:#fff,stroke-dasharray: 5 5;
    classDef test fill:#880e4f,stroke:#f48fb1,stroke-width:2px,color:#fff;
    classDef base fill:#006064,stroke:#80deea,stroke-width:1px,color:#fff;

    subgraph Application Layer
        App_Client[exe: ASR_Client]:::app
        App_Server[exe: ASR_Server]:::app
    end

    subgraph Core Library Layer
        Lib_Utils[lib: Utils]:::lib
        Lib_Network[lib: Network]:::lib
        Lib_ASR[lib: ASREngine]:::lib

        %% Inherit base settings
        Lib_Utils -. private link .- Settings[interface: arc_base_settings]:::base
        Lib_Network -. private link .- Settings
        Lib_ASR -. private link .- Settings

        %% Inter-library dependencies
        Lib_Network -- public link --> Lib_Utils
        Lib_ASR -- public link --> Lib_Utils

        %% Business library depends on third-party
        Lib_ASR -- public link --> Sherpa[shared: SherpaOnnx]:::third
        Lib_ASR -- public link --> RKNN[shared: LibRKNNRT]:::third
    end

    subgraph Dependency Layer
        RKNN
        Sherpa

        %% Corresponds to INTERFACE_LINK_LIBRARIES in sherpa-onnx/CMakeLists.txt
        Sherpa -- link --> RKNN
    end

    subgraph Test Layer
        Test_Net[exe: test_Network]:::test
        GTest[lib: GTest_Main]:::third

        Test_Net -- private link --> Lib_Network
        Test_Net -- private link --> GTest
    end

    App_Client -- link --> Lib_Network
    App_Client -- link --> Lib_ASR
    App_Client -- link --> Lib_Utils
    App_Client -- link --> Sherpa

    App_Server -- link --> Lib_Network
    App_Server -- link --> Lib_ASR
    App_Server -- link --> Lib_Utils
    App_Server -- link --> Sherpa
```

### 1.2 Design Principles
*   **Single Source of Truth**: `build.sh` acts solely as an entry point; build dependencies are entirely controlled by the CMake DAG.
*   **Out-of-Source Build**: Build artifacts are strictly isolated in the `build/` directory, preventing contamination of the source tree.
*   **DSL Driven**: Implementation modules contain minimal logic, relying solely on standard interfaces provided by `ArcFunctions`.

---

## 2. Build Lifecycle (Build Lifecycle)

This section details the entire process from executing `build.sh` to generating artifacts, illustrating the flow of control between scripts, configuration files, and CMake.

<!-- Please paste the Mermaid code for [Figure 1: ArcForge Build Process & File Interaction] here -->
<!-- Suggest using graph TB layout -->
```mermaid
graph TB
    %% ==========================================
    %% Styles Definition (Figure 1 Specific)
    %% ==========================================
    classDef entry fill:#37474f,stroke:#cfd8dc,stroke-width:2px,color:#fff;
    classDef config fill:#f57f17,stroke:#ffe082,stroke-width:2px,color:#fff;
    classDef toolchain fill:#e65100,stroke:#ffcc80,stroke-width:2px,color:#fff;
    classDef root fill:#0277bd,stroke:#81d4fa,stroke-width:2px,color:#fff;
    classDef dsl fill:#7b1fa2,stroke:#ce93d8,stroke-width:2px,color:#fff;
    classDef submodule fill:#2e7d32,stroke:#a5d6a7,stroke-width:2px,color:#fff;

    %% Avoid title overlap, adjust subgraph padding
    %% (Mermaid doesn't directly support padding, but node arrangement can mitigate this)

    %% ==========================================
    %% 1. Entry and Configuration Phase (Setup Phase)
    %% ==========================================
    subgraph Phase_Entry [Entry Phase]
        direction TB
        Entry_Script([Start: build.sh]):::entry
    end

    subgraph Phase_Config [CMakePresets.json]
        direction TB
        %% Linear logic: Define variables first, then toolchain
        Preset_Vars[1. Inject Cache Variables<br/>URLs & Hash]:::config
        Preset_Tool[2. Specify Toolchain File Path]:::config

        Preset_Vars ==> Preset_Tool
    end

    subgraph Phase_Toolchain [cmake/toolchains/xxx]
        direction TB
        %% Simplified into a single block
        TC_Env[3. Load Cross-Compilation Environment<br/>RK3588S Sysroot & Compilers]:::toolchain
    end

    %% ==========================================
    %% 2. Core Orchestration Phase (Orchestration Phase)
    %% ==========================================
    subgraph Phase_Orchestrator [Root CMakeLists.txt]
        direction TB

        %% Use Roman numerals for movements
        Orch_I[I. Load Build Functions Library<br/>include ArcFunctions]:::root
        Orch_II[II. Process Third-Party Dependencies<br/>add_subdirectory third_party]:::root
        Orch_III[III. Process Core Libraries<br/>add_subdirectory libs]:::root
        Orch_IV[IV. Process Applications<br/>add_subdirectory apps]:::root

        Orch_I ==> Orch_II ==> Orch_III ==> Orch_IV
    end

    subgraph Phase_DSL [Build Toolbox: ArcFunctions.cmake]
        direction TB
        DSL_Defs[Defines arc_* Macros & Functions]:::dsl
    end

    %% ==========================================
    %% 3. Execution Phase (Execution Phase)
    %% ==========================================
    subgraph Phase_Exec_3rd [third_party/CMakeLists.txt]
        direction TB
        Exec_3A[A. FetchContent Download/Extract]:::submodule
        Exec_3B[B. Create Imported Targets]:::submodule
        Exec_3A ==> Exec_3B
    end

    subgraph Phase_Exec_Libs [libs/CMakeLists.txt]
        direction TB
        Exec_Libs_Dist[Distribute: utils -> network -> asr]:::submodule

        subgraph Inner_Lib_Build [libs/*/CMakeLists.txt]
            Lib_Step1[A. add_library]:::submodule
            Lib_Step2[B. arc_setup_system_info]:::submodule
            Lib_Step3[C. arc_install_library]:::submodule
            Lib_Step1 ==> Lib_Step2 ==> Lib_Step3
        end

        Exec_Libs_Dist ==> Inner_Lib_Build
    end

    subgraph Phase_Exec_Apps [apps/CMakeLists.txt]
        direction TB
        Exec_Apps_Dist[Distribute: client -> server]:::submodule

        subgraph Inner_App_Build [apps/*/CMakeLists.txt]
            App_Step1[A. add_executable]:::submodule
            App_Step2[B. arc_setup_system_info]:::submodule
            App_Step3[C. arc_install_executable]:::submodule
            App_Step1 ==> App_Step2 ==> App_Step3
        end

        Exec_Apps_Dist ==> Inner_App_Build
    end

    %% ==========================================
    %% Wiring (Wiring)
    %% ==========================================

    %% 1. Entry -> Configuration
    Entry_Script ==> Preset_Vars

    %% 2. Configuration -> Toolchain Loading
    Preset_Tool -. Specifies Path .-> TC_Env

    %% 3. Environment Aggregation -> Orchestrator (Root)
    %% Toolchain Environment and Preset Variables are injected into Root
    TC_Env ==> Orch_I
    Preset_Vars -. Variable Injection .-> Orch_I

    %% 4. Orchestrator Calls DSL
    Orch_I -. include .-> DSL_Defs

    %% 5. Orchestrator Distributes Tasks (Main Flow)
    Orch_II ==> Exec_3A
    Orch_III ==> Exec_Libs_Dist
    Orch_IV ==> Exec_Apps_Dist

    %% 6. DSL Callbacks (Functions Called by Submodules)
    DSL_Defs -. function call .-> Lib_Step2
    DSL_Defs -. function call .-> Lib_Step3
    DSL_Defs -. function call .-> App_Step2
    DSL_Defs -. function call .-> App_Step3
```

**Key Phase Explanations:**
1.  **Pre-Configure (Preset)**: `CMakePresets.json` injects Toolchain and dependency version information.
2.  **Orchestration (Root)**: The root `CMakeLists.txt` acts as the conductor, loading the DSL and dispatching tasks.
3.  **Execution (Subdirs)**: Subdirectories invoke the DSL to perform specific compilation and linking.

---

## 3. Build Black Magic: ArcFunctions DSL

To simplify `CMakeLists.txt` writing, the project encapsulates `cmake/ArcFunctions.cmake`. The following diagrams illustrate the automatic operations performed when you call `arc_install_library` or `arc_setup_system_info`.

<!-- Please paste the Mermaid code for [Figure 2: ArcFunctions Core Logic (The Arc Magic)] here -->
<!-- Suggest using graph LR layout, ensuring Init/Setup/Install/Test clusters are included -->
```mermaid
graph TB
    %% ==========================================
    %% 样式定义 (图2专用)
    %% ==========================================
    classDef fileContainer fill:#263238,stroke:#90a4ae,stroke-width:2px,stroke-dasharray: 5 5,color:#fff;
    classDef initGroup fill:#1565c0,stroke:#90caf9,stroke-width:2px,color:#fff;
    classDef setupGroup fill:#ef6c00,stroke:#ffcc80,stroke-width:2px,color:#fff;
    classDef installGroup fill:#2e7d32,stroke:#a5d6a7,stroke-width:2px,color:#fff;
    classDef testGroup fill:#c2185b,stroke:#f48fb1,stroke-width:2px,color:#fff;
    classDef nodeStep fill:#455a64,stroke:#cfd8dc,stroke-width:1px,color:#fff;

    subgraph ArcFunctions_File [ArcFunctions.cmake 工具箱]
        direction TB

        %% ==========================================
        %% 1. 全局初始化簇
        %% ==========================================
        subgraph Cluster_Init [1.全局初始化类]
            direction TB

            subgraph Func_Ver [arc_extract_version_from_changelog]
                direction TB
                V1[读取 CHANGELOG.md]:::nodeStep --> V2[正则匹配 ## vX.Y.Z]:::nodeStep
                V2 --> V3[导出 ARC_PROJECT_VERSION]:::nodeStep
            end

            subgraph Func_Meta [arc_init_project_metadata]
                direction TB
                M1[获取 Version 变量]:::nodeStep --> M2[设置作者 PotterWhite]:::nodeStep
                M2 --> M3[生成 UTC 时间戳]:::nodeStep
                M3 --> M4[导出 GLOBAL__变量]:::nodeStep
            end

            subgraph Func_Global [arc_init_global_settings]
                direction TB
                G1[创建接口库 arc_base_settings]:::nodeStep
                G2[强制 C++17 & PIC]:::nodeStep --> G1
                G3[设置 -Wall -Werror]:::nodeStep --> G1
                G4[根据 Debug/Release 设优化级]:::nodeStep --> G1
                G5[导出 BUILD_TYPE]:::nodeStep --> G1
            end
        end
        class Cluster_Init initGroup

        %% ==========================================
        %% 2. 目标配置簇
        %% ==========================================
        subgraph Cluster_Setup [2.目标配置类]
            direction TB

            subgraph Func_SetupInfo [arc_setup_system_info]
                direction TB
                S1[校验命名空间]:::nodeStep --> S2[生成 system-info.h]:::nodeStep
                S2 --> S3[设置 Include 路径 src/gen]:::nodeStep
                S3 --> S4[挂载 PCH 预编译头]:::nodeStep
                S4 --> S5[设置 SOVERSION]:::nodeStep
                S5 --> S6[链接 arc_base_settings]:::nodeStep
            end

            subgraph Func_GenHeader [arc_generate_system_info_header]
                direction TB
                H1[生成通用 system-info-gen.h]:::nodeStep --> H2[创建接口库 project_version_info]:::nodeStep
            end
        end
        class Cluster_Setup setupGroup

        %% ==========================================
        %% 3. 安装部署簇
        %% ==========================================
        subgraph Cluster_Install [3.安装部署类]
            direction TB

            subgraph Func_InstLib [arc_install_library]
                direction TB
                L1[安装头文件 include/]:::nodeStep --> L2[安装二进制 lib/ bin/]:::nodeStep
                L2 --> L3[生成 ConfigVersion.cmake]:::nodeStep
                L3 --> L4[生成 Config.cmake]:::nodeStep
                L4 --> L5[导出 Targets.cmake]:::nodeStep
            end

            subgraph Func_InstExe [arc_install_executable]
                direction TB
                E1[安装 Target 到 bin/]:::nodeStep
            end
        end
        class Cluster_Install installGroup

        %% ==========================================
        %% 4. 质量保证簇
        %% ==========================================
        subgraph Cluster_Test [4.测试集成类]
            direction TB

            subgraph Func_AddTest [arc_add_test]
                direction TB
                T1[定义 exe 名称 test_Target]:::nodeStep
                T1 --> T2[链接 GTest::gtest_main]:::nodeStep
                T2 --> T3[链接 被测库 ArcForge::Target]:::nodeStep
                T3 --> T4[注入私有头文件路径 /src]:::nodeStep
                T4 --> T5[注册 gtest_discover_tests]:::nodeStep
            end
        end
        class Cluster_Test testGroup

    end
    class ArcFunctions_File fileContainer
```

**DSL's Automated Features:**
*   **Version Injection**: Automatically generates `system-info.h` with Git version and build timestamp.
*   **Standardized Installation**: Automatically handles RPATH, generates `Config.cmake` and `Targets.cmake` for standard `find_package`.
*   **Environment Isolation**: Automatically configures PCH (Precompiled Headers) and different Include paths for Build/Install phases.

---

## 4. Integration Guide (Integration Guide)

### 4.1 Internal Module Development
When developing new modules, simply call the DSL; no need to manage installation rules directly:

```cmake
# libs/new_module/CMakeLists.txt

add_library(MyModule)
# 1. Automatically configure headers, version, and alias
arc_setup_system_info(MyModule)
# 2. Link dependencies (using namespaced aliases)
target_link_libraries(MyModule PUBLIC ArcForge::Utils)
# 3. Automatically generate installation rules
arc_install_library(MyModule ${INCLUDE_DIR})
```

### 4.2 External SDK Usage
When `build/install` is packaged for distribution, third-party applications can integrate using standard CMake:

```cmake
# 1. Find the package (CMake will automatically read lib/cmake/Utils/ArcForge_UtilsConfig.cmake)
find_package(ArcForge_Utils REQUIRED)

# 2. Link (Must use the ArcForge:: prefix)
target_link_libraries(UserApp PRIVATE ArcForge::Utils)
```

---

## 5. Maintenance Command Cheatsheet

| Operation | Command | Description |
| :--- | :--- | :--- |
| **Full Build** | `./build.sh cb <plat>` | Cleans and rebuilds (Clean Build) |
| **Incremental Build** | `./build.sh build <plat>` | Compiles only modified parts, recommended for development |
| **Debug Build** | `./build.sh cb <plat> debug` | Builds with debug symbols, no optimization (-Og) |
| **Full Cleanup** | `git clean -fdx -e .env` | **Use with Caution**: Removes all untracked files (preserves .env) |