# HANDOFF — Mar30.2026 会话移交文档

> **⚠️ 已过时（Archived）**：本文件中描述的所有任务已在 2026-03-31 会话中完成。
> 具体包括：Block 1.4 ONNX 导出（export_onnx_pureBN.py）、onnx 包版本升级（1.8.1→1.14.1）、PKB 补充、docs 整理。
> 当前状态请参阅 `HANDOFF_Mar31.2026.md`。

> **写给下一个 AI Agent：**
> 这是一份会话交接 cookie。上一个 AI（Claude）因 context 即将到达上限而写下此文件。
> 你可以 100% 信任此文件，它描述了当前真实状态和你下一步要做什么。
>
> **日期**: 2026-03-30
> **移交时刻**: NEED_TO_DO 任务均未开始执行，研究阶段完成

---

## 一、我研究了什么（已完成）

本次会话只做了研究（无代码修改，无 commit），读取了以下文件：

| 文件 | 已读 |
|---|---|
| `CLAUDE.md` | ✅ |
| `docs/zh/2-progress/NEED_TO_DO.md` | ✅ |
| `docs/zh/1-for-ai/guide.md` | ✅ |
| `docs/zh/1-for-ai/codebase_map.md` | ✅ |
| `docs/zh/2-progress/progress.md` | ✅ |
| `docs/zh/2-progress/MASTER_PLAN.md` | ✅ |
| `docs/zh/2-progress/block-1.2-finetune-training.md` | ✅ |
| `docs/zh/3-highlights/architecture_vision.md` | ✅ |
| `docs/zh/1-for-ai/ai_docs_system_template.md` | ✅ |
| `docs/zh/00_INDEX.md` | ✅ |
| `docs/en/1-for-ai/guide.md`（前50行） | ✅ |

**PKB/RAG 目录扫描（只做了 find，未深入读取内容）**：
- `/home/james/syncthing/ObsidianVault/PARA-Vault/3_RESOURCE/03_Tech_Stacks_and_Domains/03.2_Artificial_Intelligence/` — 64 个文件
- `/home/james/syncthing/ObsidianVault/PARA-Vault/2_AREA/03-Area-Career/Project_ArcFoundry/` — 14 个文件
- `Project_ArcFoundry.md` 已读（概览：有 10 个待完成 task，与 helmsman 的 image matting 有关联）

---

## 二、NEED_TO_DO 任务状态（Mar30–31.2026）

**所有任务均未开始执行，0 个已完成。**

| # | 任务 | 日期组 | 状态 |
|---|---|---|---|
| 0 | 补充 helmsman 的 PKB：`/home/james/.../1_PROJECT/2025.18_Project_Helmsman` | Mar31 | ❌ 未开始 |
| 1 | PKB 整理：repo 内 plan 文档 → ObsidianVault，删除 repo 内 plan 文档 | Mar30 | ❌ 未开始 |
| 2 | docs 语言检查：en/ 不含中文，zh/ 不含英文（直译） + commit | Mar30 | ❌ 未开始 |
| 3 | Block 1.4: Export `modnet_bn_best.ckpt` → ONNX | Mar30 | ❌ 未开始 |
| 4 | Block 1.4: Verify in Netron（确认无 InstanceNormalization） | Mar30 | ❌ 未开始 |
| 5 | Block 1.4: Generate golden files `./helmsman golden` | Mar30 | ❌ 未开始 |
| 6 | Block 1.4: Build C++ native `./helmsman build cpp cb` | Mar30 | ❌ 未开始 |
| 7 | Block 1.4: Compare C++ vs Python golden with `verify_golden_tensor.py` | Mar30 | ❌ 未开始 |
| 8 | Block 1.4: Confirm visual alpha matte quality | Mar30 | ❌ 未开始 |

**用户确认的执行顺序建议**（用户的话原文）：
> "我要你先做一件事情，把你研究了那么多的研究的成果，像写成cookie或线索一样写到当前repo的docs里"
> 然后给用户看任务计划，用户自己决定下一步顺序。

---

## 三、我对各任务的研究结论

### 任务 0 — 补充 helmsman PKB（Mar31 新增任务）

**PKB 路径**：`/home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/2025.18_Project_Helmsman/`

**当前 PKB 内已有内容（本次会话已 find 扫描）**：
```
development/
├── MODNet/               ← 有 modnet的第1-4步.md，tutorial.md，terms.md
├── stage-0/              ← 早期侦察阶段，已有 AAR.md 等
├── stage-1/              ← Golden Reference 阶段笔记
└── stage-2/
    └── 任务2.4-performance-optimization/
        └── 重新训练-BN代替IN/
            └── Phrase-1_Retrain/
                ├── 1.4.md                    ← 只有一行命令（几乎空白）
                ├── Block 1.0_数据集与环境对齐.md
                ├── ✅Block 1.1 网络结构外科手术.md
                └── Block 1.2  粗调训练.md
bugs/
└── i-error_subprocess-exited-with-error.md
```

**需要做的事**（补充 PKB）：
- `Phrase-1_Retrain/1.4.md` 目前几乎空白，需要写入 Block 1.4 的详细执行计划
- 可以从 repo 的 `docs/zh/2-progress/progress.md` Block 1.4 段落提取内容
- 需要按照 PKB 已有的风格（中文，分析+步骤+成功标准格式）来写
- 这个任务和 NEED_TO_DO 任务1（PKB 整理）可以合并一起做

### 任务 1 — PKB 整理

**要做的事**：
1. 把 `docs/zh/2-progress/MASTER_PLAN.md` 内容提炼写入 PKB（项目主文件 `2025.18_Project_Helmsman.md` 的 Goals 部分，或在 development/ 下建立 roadmap.md）
2. 把 `docs/zh/2-progress/block-1.2-finetune-training.md` 内容提炼写入 PKB（`Phrase-1_Retrain/Block 1.2  粗调训练.md` 里已有这个文件，看是否要合并/完善）
3. 删除 repo 内 `docs/zh/2-progress/MASTER_PLAN.md` 和 `docs/zh/2-progress/block-1.2-finetune-training.md`
4. 同时删除 en 镜像：`docs/en/2-progress/MASTER_PLAN.md` 和 `docs/en/2-progress/block-1.2-finetune-training.md`
5. 更新 `docs/zh/00_INDEX.md` 和 `docs/en/00_INDEX.md`（删除指向 MASTER_PLAN 和 block-*.md 的行）
6. 在 `docs/zh/1-for-ai/guide.md` 中加一节 PKB RAG 路径
7. **最终向用户展示写入 PKB 的文件清单**

### 任务 2 — docs 语言检查

**已知要做的具体修复**（grep 结果分析）：
- `docs/en/` 里的 `block-1.2-finetune-training.md` 头部有中文字符（"Prerequisites" 说明块使用了中文）—— 需要逐行确认
- `docs/zh/` 里大部分标题已是中文，但需要 grep 全量确认没有遗漏的英文段落
- 检查命令：
  ```bash
  # 检查 en/ 下的中文
  grep -rn "[\u4e00-\u9fff]" docs/en/
  # 检查 zh/ 下的英文段落（代码块、命令除外）
  grep -rn "^[A-Za-z]" docs/zh/ | grep -v "^\s*\`" | grep -v "^\s*#"
  ```
- 完成后：`git add docs/ && git commit`

### 任务 3 — Block 1.4 ONNX 导出

**用户说**：卡在 ckpt → onnx。ckpt 已存到 HuggingFace：
`https://huggingface.co/PotterWhite/MODNet/tree/main`

**⚠️ 关键发现（本次会话研究结果）**：
- PKB 里 `1.4.md` 记了一个命令用 `export_onnx_pureBN_20260320` 脚本，但这个脚本 **不存在于 repo** 中
- repo 中实际可用的导出脚本是：`third-party/scripts/modnet/onnx/export_onnx_modified.py`
- 正确命令是通过 `./helmsman convert` 交互选 modified 变体，或直接运行：

```bash
source .venv/bin/activate
cd third-party/sdk/MODNet.git
python3 onnx/export_onnx_modified.py --ckpt-path checkpoints/modnet_bn_best.ckpt --output-path checkpoints/modnet_bn_best.onnx
```

**下一个 AI 的行动步骤**：
1. 检查 `.venv` 是否存在：`ls helmsman.git/.venv/`
2. 如果不存在：`cd helmsman.git && ./helmsman prepare`
3. 检查 ckpt 是否存在：`ls third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt`
4. 如果不存在，从 HuggingFace 下载（HF repo `PotterWhite/MODNet`，文件 `checkpoints/modnet_bn_best.ckpt`）
5. 运行 `./helmsman convert`（选 modified variant，选 modnet_bn_best.ckpt）
6. 导出后用 Netron 验证：**必须 0 个** `InstanceNormalization` 节点

**关键架构提醒（防止新 AI 犯错）**：
- 导出时用 `export_onnx_modified.py`，**不要用** `export_onnx.py`（后者不做 anti-fusion）
- 模型权重在 HuggingFace `PotterWhite/MODNet`，不是官方 ZHKKKe/MODNet
- Python target 版本是 **3.8.10**，不要用系统 Python
- 当前 git 分支：`retrain/modnet`（通过 `git branch` 确认）

---

## 四、关键文件路径速查

```
repo 根目录:
  /mnt/2tb_wd_purpleSurveillance_hdd/system-redirection/Development/docker/volumes/n8/src/ai/image-matting/primary-folder/helmsman.git/

PKB 路径 (RAG) — ⚠️ 重要更正：

  【主 PKB，Helmsman 专属，最重要！】
  /home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/2025.18_Project_Helmsman/
  ├── 2025.18_Project_Helmsman.md       ← 项目主文件
  ├── development/
  │   ├── MODNet/                        ← MODNet 相关笔记（modnet的第1-4步.md 等）
  │   ├── stage-0/                       ← 早期探索阶段笔记
  │   ├── stage-1/                       ← Golden Reference 阶段
  │   └── stage-2/
  │       └── 任务2.4-performance-optimization/
  │           └── 重新训练-BN代替IN/
  │               └── Phrase-1_Retrain/  ← ⭐ Block 1.0~1.4 的执行笔记在这里！
  │                   ├── 1.4.md         ← Block 1.4 已有笔记（见下方内容）
  │                   ├── Block 1.0_数据集与环境对齐.md
  │                   ├── ✅Block 1.1 网络结构外科手术.md
  │                   └── Block 1.2  粗调训练 (Fine-tuning).md
  └── bugs/

  【AI 通用知识库（技术 RAG）】
  /home/james/syncthing/ObsidianVault/PARA-Vault/3_RESOURCE/03_Tech_Stacks_and_Domains/03.2_Artificial_Intelligence/

  【ArcFoundry 项目（不同项目，但相关）】
  /home/james/syncthing/ObsidianVault/PARA-Vault/2_AREA/03-Area-Career/Project_ArcFoundry/

⭐ Block 1.4 PKB 内容发现（1.4.md 现有内容）：
  该文件目前只有一行命令，说明 Block 1.4 基本是空白的：
  python3 -m onnx.export_onnx_pureBN_20260320 ...
  ⚠️ 注意：该脚本名 export_onnx_pureBN_20260320 在 repo 里 **不存在**（已 find 确认）
  应使用：third-party/scripts/modnet/onnx/export_onnx_modified.py

核心模型文件:
  ckpt (训练完成): third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt
  目标 onnx: third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.onnx
  导出脚本（正确）: third-party/scripts/modnet/onnx/export_onnx_modified.py

Git 分支:
  当前工作分支: retrain/modnet（所有工作在这里）
  main: releases only，不要在这里工作
```

---

## 五、任务计划文档位置

**本文件** 就是任务计划和 handoff，位于：
`docs/zh/2-progress/task-logs/HANDOFF_Mar30.2026.md`

用户查看路径：打开 repo，找到 `docs/zh/2-progress/task-logs/HANDOFF_Mar30.2026.md`

---

## 六、下一个 AI 的启动协议

1. 读完本文件（你正在读）
2. 读 `docs/zh/2-progress/NEED_TO_DO.md`（确认任务列表，注意有 Mar31 和 Mar30 两个日期组）
3. 读 `docs/zh/1-for-ai/guide.md`（工作规则）
4. 询问用户想从哪个任务开始
5. **推荐顺序**：
   - 任务2（docs 语言检查，快，15分钟，有 commit）
   - 任务0+1合并（PKB 补充 helmsman + PKB 整理，中等）
   - 任务3（Block 1.4 ONNX 导出，技术核心任务）

**⚠️ 注意 NEED_TO_DO.md 里 Mar31 的新任务**：
`补充helmsman的pkb：/home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/2025.18_Project_Helmsman`
这个 PKB 路径才是 Helmsman 项目真正的 PKB，**不是** Mar30 任务1 里提到的 ArcFoundry 和 AI 资源目录。
