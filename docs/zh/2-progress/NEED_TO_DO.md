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

- [ ] arcfoundry的默认平台配置选择需要优化
目前可选择的platforms已经太多，超过8个，我需要让这里的配置文件可以按照模型先分类，不要一次性全部都堆在一起，很容易选错。
我希望可以基础按照模型本身来分个类，如果同一个模型内部都超过5个配置文件的，那可能还得再分一次类了。
你来想办法分类

- [x] 量化出错
commit: 67e7032
已修复：vision.py 新增多输入校准支持。INT8 .rknn 产出成功（5.5MB）。混合量化因 SDK 2.3.2 bug 无法执行。
详见 PKB: [[log-MR2-P4-int8-quantization]] Block 4.2

- [x] pkb需要更新
已更新：roadmap Phase-4 Block 表（4.1~4.2 ✅ + 状态/备注）、文档链接从 `⏳ 待创建` → `✅ Block 4.1~4.2 已记录`、progress.md 当前位置更新至 Phase-4、_DASHBOARD.md 时间线 + 索引 + 快速链接同步。
