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


> 三个pkb目录：
> 1. /development/volumes_pkb_ai
> 2. /development/volumes_pkb_arcfoundry
> 3. /development/volumes_pkb_helmsman

---

- [x] PKB 整理：Phase 重命名 + 路线图融合 + 目录重组
    1. 00-Phase总览 融合进「任务 2.4 集成、测量与最终验证.md」（设计上的 roadmap 文件），原文件删除
    2. Phase-1/ → Phase-1-Retrain/，Phase-1-Again/ 内容迁移进 Phase-1-Retrain/Phase-1-Again-Docs/（内容融合，未删除）
    3. Phase-RKNN-PureBN/ → Phase-3-RKNN-PureBN/（加数字 prefix，符合 Phase-N- 规则）
    4. Phase-2-GF 链接引用在相关文档中修正（Phase-3-GF 旧引用 → Phase-2-GF）
    5. roadmap.md 更新当前状态为 Phase-3-RKNN-PureBN 待做
    6. Phase-3-Distill（空目录）删除
    commit: (pending)
