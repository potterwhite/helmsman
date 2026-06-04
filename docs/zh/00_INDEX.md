# Helmsman — 文档索引

> 最后更新：2026-05-22 · 状态：MR2 Phase-5 进行中（Frontend 重构 + Pipeline 初始化统一）

---

## 1 · AI Agent 文档 (`1-for-ai/`)

每次会话开始时先读这里，读完再碰任何代码。

| 文档 | 用途 |
|---|---|
| [`1-for-ai/README.md`](1-for-ai/README.md) | AI 文档索引入口 |
| [`1-for-ai/architecture.md`](1-for-ai/architecture.md) | 整体架构（Pipeline → Frontend/Engine/BackEnd → Mode） |
| [`1-for-ai/frontend.md`](1-for-ai/frontend.md) | Frontend 模块详解（子类、子阶段、数据流） |
| [`1-for-ai/build-system.md`](1-for-ai/build-system.md) | CMake 构建系统（CMAKE_PLATFORM、条件编译） |
| [`1-for-ai/data-flow.md`](1-for-ai/data-flow.md) | 端到端数据流（mp4 → demux → decode → infer → composite） |
| [`1-for-ai/timing.md`](1-for-ai/timing.md) | 计时系统详解（ScopedTimer、StageAccumulator） |
| [`1-for-ai/naming-convention.md`](1-for-ai/naming-convention.md) | 命名规范速查（Google C++ Style） |
| [`1-for-ai/guide.md`](1-for-ai/guide.md) | 工作规则、commit 格式、工作流、常见陷阱 |
| [`1-for-ai/codebase_map.md`](1-for-ai/codebase_map.md) | 完整代码库结构——用此代替扫描源文件 |
| [`1-for-ai/ai_docs_system_template.md`](1-for-ai/ai_docs_system_template.md) | 本 docs 系统的建设规范与模板 |

---

## 2 · 项目进度 (`2-progress/`)

| 文档 | 用途 |
|---|---|
| [`2-progress/progress.md`](2-progress/progress.md) | 所有 Phase/Block 详情、当前状态、性能基准、架构决策日志 |
| [`2-progress/NEED_TO_DO.md`](2-progress/NEED_TO_DO.md) | 活跃任务待办列表（checkbox 格式，最新在顶部） |
