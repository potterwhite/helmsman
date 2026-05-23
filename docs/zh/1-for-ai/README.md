# AI 代码库文档索引

本目录包含供 Claude Code 阅读的代码库架构文档。**不要直接扫描代码库，先阅读本文档。**

## 文档列表

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | 整体架构（Pipeline → Frontend/Engine/Backend → Mode） |
| [frontend.md](frontend.md) | Frontend 模块详解（子类、子阶段、数据流） |
| [build-system.md](build-system.md) | CMake 构建系统（CMAKE_PLATFORM、工具链、条件编译） |
| [data-flow.md](data-flow.md) | 端到端数据流（mp4 → demux → decode → preprocess → infer → composite → output） |
| [timing.md](timing.md) | 计时系统详解（ScopedTimer、StageAccumulator、嵌套关系） |
| [naming-convention.md](naming-convention.md) | 命名规范（Google C++ Style 速查） |

## 更新原则

- 架构设计决策和理由写在这里
- 具体代码实现不写（代码本身就是文档）
- 代码改变时记得更新对应文档
