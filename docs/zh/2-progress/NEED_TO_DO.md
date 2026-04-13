# NEED_TO_DO — 当前待办

> **AI Agent 操作规则（必读，每次都要执行）：**
>
> 完成一个 checkbox 后，立即执行以下两步，缺一不可：
> 1. 把该条目的 `[ ]` 改为 `[x]`，并在其下一行追加 `commit: <hash>`
> 2. 把整个已完成的日期组（含所有 `[x]` 条目）**拆散归档到 PKB**：
>    - 按内容归属散入 PKB 对应阶段目录（foundation/、model/round1-modnet/、model/round2-rvm/、deploy/ 等）
>    - 定位为 bug-/log- 类文件，**不能放 tasks/ 目录**（Obsidian plugin 保留）
>    - 如果可以追加到现有 log/bug 文件则追加；否则新建文件
>    - 纯 housekeeping / meta 条目 → `foundation/log-F-misc-session-notes.md`
>    - 确保每个归档条目都能从 `_DASHBOARD.md` 时间线中触达
>
> **PKB 路径**：`/development/volumes_pkb_helmsman/`
>
> 本文件归档后只保留**未完成**的条目。已完成记录永久保存在 PKB 里。


> 三个pkb目录：
> 1. /development/volumes_pkb_ai
> 2. /development/volumes_pkb_arcfoundry
> 3. /development/volumes_pkb_helmsman

> 开发板ssh
> ssh evboard
> 密码：123
> 主要工作目录： /root/
> 命令举例：Helmsman_Matting_Client /root/media/green-fall-girl-point-to_padded_1920x1080.png /root/modnet_pureBN_fp16.rknn /root/debug/cpp/

---

*(当前无未完成条目 — 2026-04-13 所有 housekeeping 已归档至 PKB)*