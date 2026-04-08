# AI-First Docs 系统 — 可移植模板

> **模板用途（AI agent 必读）：**
> 该模板是为了让 AI agent 能在任何新的项目仓库下，无任何外部辅助信息的情况下，
> 直接原样 100% 复刻本 docs 系统。
> 照着本文件的结构和规范建立文档，后续每个 AI agent 会话都能在 3-4 个文件内完成
> 定向，无需扫描源代码。

---

## 一、设计意图

| 不用这套系统时 | 用这套系统后 |
|---|---|
| AI 每次扫描 50-200 个源文件来理解项目 | AI 读 3-4 个文档，直接写代码 |
| 40-60% token 用在"读懂项目"上 | >90% token 用在实际任务上 |
| 各 AI 自己发明模式，前后不一致 | 所有 AI 遵循同一套规范 |
| 跨会话遗忘约定 | 每次会话开头重读，几秒恢复上下文 |

启动成本：约 2000-4000 token。
节省：每次会话 2-8 万 token（消除源码扫描）。
第一个任务起就回本。

---

## 二、完整目录结构

```
<repo-root>/
├── CLAUDE.md                          ← Claude Code 自动注入；会话入口；必须 ≤60 行
└── docs/
    ├── zh/                            ← 主语言（中文）；所有内容以此为准
    │   ├── 00_INDEX.md                ← 导航中枢；人类 + AI 一张表找到所有文档
    │   ├── 1-for-ai/                  ← AI agent 专用文档
    │   │   ├── guide.md               ← 工作规则 · commit 格式 · 架构事实
    │   │   ├── codebase_map.md        ← 代码库地图（代替扫描源文件）
    │   │   └── ai_docs_system_template.md  ← 本文件；docs 系统建设指南
    │   ├── 2-progress/                ← 进度跟踪
    │   │   ├── progress.md            ← Phase/Block 详情 · 当前状态 · 决策日志
    │   │   ├── NEED_TO_DO.md          ← 活跃待办（checkbox；**只在 zh 里维护**）
    │   │   ├── MASTER_PLAN.md         ← 项目路线图
    │   │   └── task-logs/             ← 已完成条目归档目录
    │   │       └── NEED_TO_DO_ARCHIVED_<MonDD>.<YYYY>.md
    │   ├── 3-highlights/              ← 架构亮点 · 设计决策
    │   │   └── architecture_vision.md
    │   ├── 4-for-beginer/             ← README · quick-start（面向用户，不面向 AI）
    │   │   └── README_ZH_CN.md
    │   └── assets/                    ← zh 专用图片（浅色主题背景等）
    └── en/                            ← 英文镜像（zh 的一一对应翻译）
        ├── 00_INDEX.md
        ├── 1-for-ai/
        │   ├── guide.md
        │   └── codebase_map.md
        ├── 2-progress/
        │   ├── progress.md
        │   ├── MASTER_PLAN.md
        │   └── block-*.md
        ├── 3-highlights/
        │   └── architecture_vision.md
        └── assets/                    ← en 专用图片（深色主题背景等）
```

**关键约定：**
- `zh/` 是唯一的权威来源，`en/` 是 `zh/` 的翻译镜像
- `NEED_TO_DO.md` **只在 zh 下维护**，en 不建立对应文件
- `task-logs/` **只在 zh/2-progress/` 下**，en 不建立对应目录

---

## 三、文件读取频率

| 文件 | 何时读 | 用途 |
|---|---|---|
| `CLAUDE.md` | 每次 turn 自动注入 | 入口；必须 ≤60 行 |
| `guide.md` | 每次会话读一次 | 规则 + 架构事实 |
| `codebase_map.md` | 每次会话读一次 | 结构 + 文件查询 |
| `progress.md` | 每次会话读一次 | 当前阶段上下文 |
| `NEED_TO_DO.md` | 处理任务时读 | 活跃待办列表 |
| `00_INDEX.md` | 需要导航时读 | 所有文档目录 |
| 深度参考文档 | 任务需要时读 | API spec、领域规则等 |

---

## 四、各文件规范

### 4.1 CLAUDE.md — 会话入口

Claude Code 每个 turn 自动注入此文件。**必须 ≤60 行**，所有细节放到它指向的文档里。

四个必要 section：
1. **Session Start Protocol** — 按顺序列出要读的文件；明确写 "不要扫描 `<源码目录>/`"
2. **Requirements** — 语言要求、沟通风格
3. **Commands** — 项目关键 CLI 命令
4. **Documentation Map** — "需求 → 文件" 一行一行的对照表

```markdown
# CLAUDE.md

This file provides essential guidance to Claude Code when working in this repository.
**For full context, always start by reading the docs** (see below).

## ⚠️ Session Start Protocol

**Before writing any code**, read these files in order:
1. `docs/zh/2-progress/NEED_TO_DO.md` — 当前紧急任务
2. `docs/zh/1-for-ai/guide.md` — 工作规则、commit 格式、关键架构事实
3. `docs/zh/1-for-ai/codebase_map.md` — 完整代码库地图（不要自己扫描）

Then check `docs/zh/2-progress/progress.md` for current phase status.

Do **not** scan `<primary-source-dir>/` before reading the above.

---

## Requirements

- All source code and comments must be in **English**
- Communicate with me in **中文**
- Do not end the session — always wait for next instruction via AskUserQuestion

---

## Commands

<粘贴项目关键 CLI 命令>

---

## Documentation Map

| Need | File |
|---|---|
| Working rules + commit format | `docs/zh/1-for-ai/guide.md` |
| Codebase structure | `docs/zh/1-for-ai/codebase_map.md` |
| Phase progress + roadmap | `docs/zh/2-progress/progress.md` |
| Active task backlog | `docs/zh/2-progress/NEED_TO_DO.md` |
| Architecture vision | `docs/zh/3-highlights/architecture_vision.md` |
| All docs index | `docs/zh/00_INDEX.md` |
```

---

### 4.2 NEED_TO_DO.md — 活跃待办

**只在 `docs/zh/2-progress/NEED_TO_DO.md` 维护，en 无对应文件。**

文件顶部必须包含如下 AI 操作规则块（逐字复制，不要修改措辞）：

```markdown
# NEED_TO_DO — 当前待办

> **AI Agent 操作规则（必读，每次都要执行）：**
>
> 完成一个 checkbox 后，立即执行以下两步，缺一不可：
> 1. 把该条目的 `[ ]` 改为 `[x]`，并在其下一行追加 `commit: <hash>`
> 2. 把整个已完成的日期组（含所有 `[x]` 条目）**剪切**（从本文件删除），**追加**到
>    `task-logs/NEED_TO_DO_ARCHIVED_<MonDD>.<YYYY>.md` 的末尾
>    （文件不存在则新建；同一天多次归档就不断 append 到同一文件）
>
> 本文件归档后只保留**未完成**的条目。已完成记录永久保存在 `task-logs/` 里。
```

**条目格式：**

```markdown
<Mon><DD>.<YYYY> <HH:MM>

- [ ] <待办描述>
- [ ] <待办描述>
    <可选：补充说明，缩进 4 格>
```

**规则：**
- 最新日期组在文件**最顶部**（顶部 = 最新）
- 一个工作段 = 一个日期组（同一天开多次会话就写多个日期组）
- 完成后**整个日期组**（包括日期行和所有条目）都要剪切走，不留残影
- `task-logs/` 里的归档文件才是历史记录，**永远不删除**

---

### 4.3 task-logs/ — 归档目录

位置：`docs/zh/2-progress/task-logs/`

文件命名：`NEED_TO_DO_ARCHIVED_<MonDD>.<YYYY>.md`（例：`NEED_TO_DO_ARCHIVED_Mar30.2026.md`）

归档文件格式（不断 append，不覆盖）：

```markdown
# NEED_TO_DO 归档 — <MonDD>.<YYYY>

---

## <Mon><DD>.<YYYY> <HH:MM>

- [x] <已完成条目>
- [x] <已完成条目>

commit: `<hash>`

---

## <Mon><DD>.<YYYY> <HH:MM>   ← 同一天第二个归档 append 在这里

- [x] <已完成条目>

commit: `<hash>`
```

---

### 4.4 zh/00_INDEX.md — 导航中枢

内容只有表格，不放实质内容。每个文件一行，含相对链接和一句话说明用途。

```markdown
# <项目名> — 文档索引

> 最后更新：<date> · 状态：<当前阶段一句话>
> **English →** [../en/00_INDEX.md](../en/00_INDEX.md)

---

## 1 · AI Agent & 开发者文档 (`1-for-ai/`)

| 文档 | 用途 |
|---|---|
| [`1-for-ai/guide.md`](1-for-ai/guide.md) | ⭐ 工作规则、commit 格式、关键架构事实 |
| [`1-for-ai/codebase_map.md`](1-for-ai/codebase_map.md) | ⭐ 完整代码库结构——代替扫描源文件 |

---

## 2 · 项目进度 (`2-progress/`)

| 文档 | 用途 |
|---|---|
| [`2-progress/progress.md`](2-progress/progress.md) | Phase/Block 详情、性能基准、决策日志 |
| [`2-progress/NEED_TO_DO.md`](2-progress/NEED_TO_DO.md) | 活跃待办（checkbox，最新在顶） |

---

## 3 · 架构亮点 (`3-highlights/`)

| 文档 | 用途 |
|---|---|
| [`3-highlights/architecture_vision.md`](3-highlights/architecture_vision.md) | 设计哲学和关键决策 |

---

## 4 · 入门文档 (`4-for-beginer/`)

| 文档 | 用途 |
|---|---|
| [`4-for-beginer/README_ZH_CN.md`](4-for-beginer/README_ZH_CN.md) | 中文 README、快速上手 |
```

---

### 4.5 en/*.md — 英文镜像规则

- 每个 `en/*.md` 对应一个 `zh/*.md`，内容是其完整翻译
- `en/*.md` 顶部的第一个 blockquote 里，必须加一行：
  `> **中文版 →** [对应 zh 文件的相对路径](相对路径)`
- `zh/*.md` 顶部对应加：
  `> **English →** [对应 en 文件的相对路径](相对路径)`
- `NEED_TO_DO.md` 和 `task-logs/` **不建立 en 镜像**

---

### 4.6 guide.md — 工作规则

AI 每次会话读一次。包含以下七个 section（顺序固定）：

1. **Reading Order** — 会话开始后按顺序读哪些文件
2. **Non-Negotiable Rules** — 语言、commit、文档维护
3. **Commit Message Format** — 格式模板 + 举例
4. **How to Handle Human Requests** — 新功能 / bug / 重构的处理流程
5. **Common Pitfalls** — 项目特有的常见错误（表格形式，≥5 条）
6. **Key Architecture Facts** — AI 最容易犯错的架构事实（5-10 条）
7. **Development Commands** — 与 CLAUDE.md 保持一致（冗余是有意为之）

---

### 4.7 codebase_map.md — 代码库地图

**这是最重要的文件。** 写好它，AI 不需要打开任何源文件就能行动。

必须包含：

1. **警告 header** + 维护规则 + Last updated（含原因）
2. **仓库根目录 ASCII 树**
3. **每个非平凡模块的逐文件说明**（包含函数名、关键参数、行为描述）
4. **关键架构模式**（编号列表，每条 1-2 句）

Header 模板（逐字复制）：

```markdown
# <项目名> — Codebase Map (AI Agent Quick Reference)

> **⚠️ FOR AI AGENTS — READ THIS FIRST**
> This document is the single source of truth for codebase structure.
> **Do NOT do a full repo scan** — read this file instead.
>
> **Maintenance rule:** Any AI agent that modifies a file listed here MUST update
> the relevant section in this document in the same commit/session.
>
> Last updated: <date> (<一行原因，例如 "added port_calc.sh section">)
>
> **English →** [../../en/1-for-ai/codebase_map.md](../../en/1-for-ai/codebase_map.md)
```

> **为什么把维护规则放在 codebase_map header 里？**
> AI 在每次编码任务前都会读 codebase_map。如果规则只写在 CLAUDE.md 或 guide.md，
> AI 写完代码后可能已经忘了。放在 codebase_map 顶部是 AI 改文件前最后读到的提醒。

---

### 4.8 assets/ 规则

- `docs/zh/assets/` — 放浅色/中文主题图片（如项目 banner 浅色版）
- `docs/en/assets/` — 放深色/英文主题图片（如项目 banner 深色版）
- 每个 `assets/` 只放属于自己语言版本的图片，不共享
- README 里引用图片时：根目录 README.md（英文）引用 `docs/en/assets/`；
  zh README 引用 `docs/zh/assets/`

---

## 五、在新项目里建立此系统——核查清单

```
[ ] 在仓库根目录创建 CLAUDE.md
      - Session start protocol，按顺序列出 guide → codebase_map → progress → NEED_TO_DO
      - 明确写 "Do not scan <具体源码目录>/"（写真实路径，不要写 src/ 这种模糊说法）
      - 项目关键 CLI 命令
      - Documentation Map 表格（Need → File）
      - 保持 ≤60 行

[ ] 创建 docs/zh/00_INDEX.md
      - 三个表格：1-for-ai / 2-progress / 3-highlights / 4-for-beginer
      - 每行一个文件，含相对链接和一句话用途
      - 顶部加 English → en/00_INDEX.md 链接

[ ] 创建 docs/zh/1-for-ai/guide.md
      - 7 个 section 按顺序
      - 读取顺序里的 NEED_TO_DO 条目注明 "only if you have tasks to do"
      - Common Pitfalls ≥5 条，全部项目特有（不要写通用废话）
      - Key Architecture Facts 5-10 条，专门针对 AI 最容易搞错的地方

[ ] 创建 docs/zh/1-for-ai/codebase_map.md
      - 逐字复制 header 模板（含维护规则）
      - Last updated 行必须写原因
      - ASCII 树 + 每个非平凡文件的描述（够详细到 AI 不用打开文件就能行动）
      - 末尾写架构模式编号列表

[ ] 创建 docs/zh/2-progress/progress.md
      - 总体状态表（所有 Phase，emoji 状态）
      - "Currently active: Phase X Block Y" 一行
      - 每个已完成和进行中 Phase 的详情
      - 已完成步骤带 commit hash（防止 AI 重复做）

[ ] 创建 docs/zh/2-progress/NEED_TO_DO.md
      - 逐字复制顶部 AI 操作规则块（不要修改措辞）
      - 只放未完成条目
      - 最新日期组在最顶部

[ ] 创建 docs/zh/2-progress/task-logs/ 目录（空目录可放 .gitkeep）

[ ] 创建 docs/zh/3-highlights/architecture_vision.md
      - 解释项目为什么这样建
      - 关键设计决策及其理由

[ ] 为每个 zh 文件创建对应的 en 镜像文件
      - en/*.md 顶部加 "中文版 →" 链接
      - zh/*.md 顶部加 "English →" 链接
      - NEED_TO_DO.md 和 task-logs/ 不建 en 镜像

[ ] docs/zh/assets/ 和 docs/en/assets/ 分别放各自语言版本的图片

[ ] 检查 CLAUDE.md 里的路径全部指向 docs/zh/（不要指向 docs/architecture/ 等旧路径）
```

---

## 六、常见错误

| ❌ 错误 | ✅ 正确做法 |
|---|---|
| CLAUDE.md 超过 100 行 | 保持 ≤60 行；细节放 guide.md |
| codebase_map 条目只写一句 "Contains API views" | 写函数名、关键参数、行为；让 AI 不用开文件就能行动 |
| `Last updated` 只写日期不写原因 | 始终加原因：`(added X, updated Y to Z)` |
| 把已完成条目留在 NEED_TO_DO.md 里 | 完成后立即剪切到 task-logs/ 归档 |
| 忘记归档就直接改 NEED_TO_DO | 先归档（剪切+写入 task-logs），再更新文件 |
| en 也建了 NEED_TO_DO.md | NEED_TO_DO 只在 zh 维护 |
| zh 和 en 的 assets 放在同一目录 | 各放各的：zh/assets/ 和 en/assets/ |
| README 背景图引用了不存在的路径 | en README → docs/en/assets/；zh README → docs/zh/assets/ |
| zh README 的 "English" 链接深度错误 | 从文件实际位置算相对路径，不要凭感觉写 |
| 把已删除的旧路径留在 CLAUDE.md 里 | 每次重组后立即校对 CLAUDE.md 里所有路径 |
| 删除了已归档的历史条目 | task-logs/ 里的文件只 append，永远不删除 |
