# ArcForge 构建系统详解

本文档旨在揭示 ArcForge 项目的构建系统架构。项目采用 **"Modern CMake"** 理念，通过高度封装的 DSL（领域特定语言）实现了构建逻辑的自动化与标准化。

---

## 1. 核心架构视图

### 1.1 模块依赖拓扑 (Module Topology)
ArcForge 采用严格的分层架构。底层模块对上层不可见，上层通过接口继承底层功能。

<!-- 请在此处粘贴 [图三：模块依赖架构图 (Dependency Graph)] 的 Mermaid 代码 -->
<!-- 建议使用 graph BT 或 LR 布局 -->
```mermaid
graph BT
    %% ==========================================
    %% 样式定义 (图3专用)
    %% ==========================================
    classDef app fill:#0d47a1,stroke:#90caf9,stroke-width:2px,color:#fff;
    classDef lib fill:#e65100,stroke:#ffcc80,stroke-width:2px,color:#fff;
    classDef third fill:#333333,stroke:#ffffff,stroke-width:2px,color:#fff,stroke-dasharray: 5 5;
    classDef test fill:#880e4f,stroke:#f48fb1,stroke-width:2px,color:#fff;
    classDef base fill:#006064,stroke:#80deea,stroke-width:1px,color:#fff;

    subgraph 基础配置层
        Settings[interface: arc_base_settings]:::base
    end

    subgraph 第三方依赖层
        RKNN[shared: LibRKNNRT]:::third
        Sherpa[shared: SherpaOnnx]:::third

        %% 对应 sherpa-onnx/CMakeLists.txt 中的 INTERFACE_LINK_LIBRARIES
        Sherpa -- link --> RKNN
    end

    subgraph 核心库层
        Lib_Utils[lib: Utils]:::lib
        Lib_Network[lib: Network]:::lib
        Lib_ASR[lib: ASREngine]:::lib

        %% 继承基础配置
        Lib_Utils -. private link .- Settings
        Lib_Network -. private link .- Settings
        Lib_ASR -. private link .- Settings

        %% 库间依赖
        Lib_Network -- public link --> Lib_Utils
        Lib_ASR -- public link --> Lib_Utils

        %% 业务库依赖第三方
        Lib_ASR -- public link --> Sherpa
        Lib_ASR -- public link --> RKNN
    end

    subgraph 应用层
        App_Client[exe: ASR_Client]:::app
        App_Server[exe: ASR_Server]:::app

        App_Client -- link --> Lib_Network
        App_Client -- link --> Lib_ASR
        App_Client -- link --> Lib_Utils
        App_Client -- link --> Sherpa

        App_Server -- link --> Lib_Network
        App_Server -- link --> Lib_ASR
        App_Server -- link --> Lib_Utils
        App_Server -- link --> Sherpa
    end

    subgraph 测试层
        Test_Net[exe: test_Network]:::test
        GTest[lib: GTest_Main]:::third

        Test_Net -- private link --> Lib_Network
        Test_Net -- private link --> GTest
    end
```

### 1.2 设计原则
*   **单一事实来源**：`build.sh` 仅作入口，构建依赖关系完全由 CMake DAG 控制。
*   **源码外构建**：构建产物严格隔离在 `build/` 目录，禁止污染源码树。
*   **DSL 驱动**：业务模块（Implementation）不包含复杂逻辑，仅调用 `ArcFunctions` 提供的标准接口。

---

## 2. 构建生命周期 (Build Lifecycle)

从执行 `build.sh` 到产物生成，控制权在脚本、配置文件与 CMake 之间流转的全过程。

<!-- 请在此处粘贴 [图一：ArcForge 构建流程图 (Build Process & File Interaction)] 的 Mermaid 代码 -->
<!-- 建议使用 graph TB 布局 -->
```mermaid
graph TB
    %% ==========================================
    %% 样式定义 (图1专用)
    %% ==========================================
    classDef entry fill:#37474f,stroke:#cfd8dc,stroke-width:2px,color:#fff;
    classDef config fill:#f57f17,stroke:#ffe082,stroke-width:2px,color:#fff;
    classDef toolchain fill:#e65100,stroke:#ffcc80,stroke-width:2px,color:#fff;
    classDef root fill:#0277bd,stroke:#81d4fa,stroke-width:2px,color:#fff;
    classDef dsl fill:#7b1fa2,stroke:#ce93d8,stroke-width:2px,color:#fff;
    classDef submodule fill:#2e7d32,stroke:#a5d6a7,stroke-width:2px,color:#fff;

    %% 避免标题遮挡，调整子图内部边距
    %% (Mermaid不直接支持padding，但通过节点排布可以缓解)

    %% ==========================================
    %% 1. 入口与配置阶段 (Setup Phase)
    %% ==========================================
    subgraph Phase_Entry [入口阶段]
        direction TB
        Entry_Script([Start: build.sh]):::entry
    end

    subgraph Phase_Config [CMakePresets.json]
        direction TB
        %% 线性逻辑：先定变量，再定工具链
        Preset_Vars[1.注入 Cache 变量<br/>URLs & Hash]:::config
        Preset_Tool[2.指定 Toolchain 文件路径]:::config

        Preset_Vars ==> Preset_Tool
    end

    subgraph Phase_Toolchain [cmake/toolchains/xxx]
        direction TB
        %% 简化为一个块
        TC_Env[3.加载交叉编译环境<br/>RK3588S Sysroot & Compilers]:::toolchain
    end

    %% ==========================================
    %% 2. 核心编排阶段 (Orchestration Phase)
    %% ==========================================
    subgraph Phase_Orchestrator [Root CMakeLists.txt]
        direction TB

        %% 使用罗马数字标记乐章
        Orch_I[I. 加载构建函数库<br/>include ArcFunctions]:::root
        Orch_II[II. 处理第三方依赖<br/>add_subdirectory third_party]:::root
        Orch_III[III. 处理核心库<br/>add_subdirectory libs]:::root
        Orch_IV[IV. 处理应用程序<br/>add_subdirectory apps]:::root

        Orch_I ==> Orch_II ==> Orch_III ==> Orch_IV
    end

    subgraph Phase_DSL [构建工具箱: ArcFunctions.cmake]
        direction TB
        DSL_Defs[定义 arc_* 宏与函数]:::dsl
    end

    %% ==========================================
    %% 3. 执行落地阶段 (Execution Phase)
    %% ==========================================
    subgraph Phase_Exec_3rd [third_party/CMakeLists.txt]
        direction TB
        Exec_3A[A. FetchContent 下载/解压]:::submodule
        Exec_3B[B. 创建 Imported Targets]:::submodule
        Exec_3A ==> Exec_3B
    end

    subgraph Phase_Exec_Libs [libs/CMakeLists.txt]
        direction TB
        Exec_Libs_Dist[分发: utils -> network -> asr]:::submodule

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
        Exec_Apps_Dist[分发: client -> server]:::submodule

        subgraph Inner_App_Build [apps/*/CMakeLists.txt]
            App_Step1[A. add_executable]:::submodule
            App_Step2[B. arc_setup_system_info]:::submodule
            App_Step3[C. arc_install_executable]:::submodule
            App_Step1 ==> App_Step2 ==> App_Step3
        end

        Exec_Apps_Dist ==> Inner_App_Build
    end

    %% ==========================================
    %% 连线关系 (Wiring)
    %% ==========================================

    %% 1. 入口 -> 配置
    Entry_Script ==> Preset_Vars

    %% 2. 配置 -> 工具链加载
    Preset_Tool -. 指定路径 .-> TC_Env

    %% 3. 环境汇聚 -> 指挥中心 (Root)
    %% 工具链环境 和 预设变量 都注入到 Root
    TC_Env ==> Orch_I
    Preset_Vars -. 变量注入 .-> Orch_I

    %% 4. 指挥中心调用 DSL
    Orch_I -. include .-> DSL_Defs

    %% 5. 指挥中心分发任务 (主流程)
    Orch_II ==> Exec_3A
    Orch_III ==> Exec_Libs_Dist
    Orch_IV ==> Exec_Apps_Dist

    %% 6. DSL 回调 (函数被子模块调用)
    DSL_Defs -. function call .-> Lib_Step2
    DSL_Defs -. function call .-> Lib_Step3
    DSL_Defs -. function call .-> App_Step2
    DSL_Defs -. function call .-> App_Step3
```

**关键阶段说明：**
1.  **Pre-Configure (Preset)**：`CMakePresets.json` 注入工具链（Toolchain）与依赖版本信息。
2.  **Orchestration (Root)**：根目录 `CMakeLists.txt` 作为指挥中心，加载 DSL 并分发任务。
3.  **Execution (Subdirs)**：子目录调用 DSL 完成具体的编译与链接。

---

## 3. 构建黑科技：ArcFunctions DSL

为了简化 `CMakeLists.txt` 的编写，项目封装了 `cmake/ArcFunctions.cmake`。以下图表展示了当你调用 `arc_install_library` 或 `arc_setup_system_info` 时，系统内部自动执行的操作。

<!-- 请在此处粘贴 [图二：ArcFunctions 核心逻辑 (The Arc Magic)] 的 Mermaid 代码 -->
<!-- 建议使用 graph LR 布局，确保包含 Init/Setup/Install/Test 四个簇 -->
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

**DSL 带来的自动化特性：**
*   **版本注入**：自动生成 `system-info.h`，包含 Git 版本与构建时间戳。
*   **安装标准化**：自动处理 RPATH，生成 `Config.cmake` 和 `Targets.cmake`，支持标准 `find_package`。
*   **环境隔离**：自动配置 PCH（预编译头）和 Build/Install 阶段不同的 Include 路径。

---

## 4. 集成指南 (Integration Guide)

### 4.1 内部模块开发
新增模块时，无需关心安装规则，只需调用 DSL：

```cmake
# libs/new_module/CMakeLists.txt

add_library(MyModule)
# 1. 自动配置头文件、版本、别名
arc_setup_system_info(MyModule)
# 2. 链接依赖 (使用带命名空间的别名)
target_link_libraries(MyModule PUBLIC ArcForge::Utils)
# 3. 自动生成安装规则
arc_install_library(MyModule ${INCLUDE_DIR})
```

### 4.2 外部 SDK 使用
当 `build/install` 被打包发布后，第三方应用可通过标准 CMake 方式集成：

```cmake
# 1. 查找包 (CMake 会自动读取 lib/cmake/Utils/ArcForge_UtilsConfig.cmake)
find_package(ArcForge_Utils REQUIRED)

# 2. 链接 (必须使用 ArcForge:: 前缀)
target_link_libraries(UserApp PRIVATE ArcForge::Utils)
```

---

## 5. 维护命令速查

| 操作 | 命令 | 说明 |
| :--- | :--- | :--- |
| **全量构建** | `./build.sh cb <plat>` | 清理并重新构建 (Clean Build) |
| **增量构建** | `./build.sh build <plat>` | 仅编译修改部分，开发推荐 |
| **调试版本** | `./build.sh cb <plat> debug` | 带符号表，无优化 (-Og) |
| **彻底清理** | `git clean -fdx -e .env` | **危险**：删除所有未跟踪文件 (保留 .env) |

