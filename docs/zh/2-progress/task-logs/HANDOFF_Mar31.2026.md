# HANDOFF — Mar31.2026 会话移交文档

> **写给下一个 AI Agent**
> **日期**：2026-03-31
> **Git 分支**：`retrain/modnet`（所有工作在此，`main` 仅用于发版）

---

## 一、今天完成了什么

| 成果 | 详情 |
|---|---|
| `export_onnx_pureBN.py` 已创建 | `third-party/scripts/modnet/onnx/`，符号链接到 `MODNet.git/onnx/` |
| ONNX 导出成功 | `modnet_bn_best.ckpt` → `modnet_bn_best_pureBN.onnx`（25M，opset 11）|
| 节点验证通过 | 0 个 `InstanceNormalization` 节点；BN fold 进 Conv（正常） |

**修复的三个 Bug**：
1. `func_3_3_rebuild_sdk`：submodule 未初始化时 reset 到错误 HEAD（commit `1a88cbe`）
2. `onnx==1.8.1` 不兼容 PyTorch 2.0.1 → 升级至 `onnx==1.14.1`，`onnxruntime==1.6.0` → `1.15.1`（commit `554ca64`）
3. 本地 `MODNet.git/onnx/` 目录遮蔽 pip `onnx` 包 → `export_onnx_pureBN.py` 内含 sys.path 手术（commit `554ca64`）

---

## 二、Block 1.4 当前状态

```
✅ ONNX 导出 + 节点验证
🔜 Golden 文件生成 + C++ 对比（等测试图片）
```

**剩余步骤**（按顺序执行）：

```bash
# 1. 把测试图片（jpg/png）放入 helmsman.git/media/

# 2. 生成 golden 文件（交互选 modnet_bn_best_pureBN.onnx + 测试图片）
./helmsman golden

# 3. C++ native build
./helmsman build cpp cb

# 4. C++ 推理
runtime/cpp/install/native/release/bin/Helmsman_Matting_Client \
    <img> checkpoints/modnet_bn_best_pureBN.onnx <output_dir>

# 5. 对比 Python golden vs C++ 输出
python3 tools/MODNet/verify_golden_tensor.py
```

**成功标准**：C++ 与 Python golden 浮点误差范围内一致；视觉遮罩边缘清晰、发丝可见。

---

## 三、Block 1.4 完成后——下一大阶段

**Phase 2 INT8 量化**（必须在 docker container 里做）：

| 步骤 | 内容 |
|---|---|
| 2.1 | 从 P3M-10k 抽 200 张图作 RKNN Toolkit 2 校准集 |
| 2.2 | `modnet_bn_best_pureBN.onnx` → `modnet_bn_best_pureBN.rknn`（asymmetric INT8） |
| 2.3 | 板端测试目标：< 25ms @ 256×256 on RK3588 NPU |

> ⚠️ **环境切换提示**：Phase 2 必须在 **docker container** 里操作（有完整 RKNN Toolkit 2 工具链）。
> host ubuntu 上只能做 ONNX 导出/验证（今天已完成）。

---

## 四、AI 角色提醒（必读）

**AI 是教练，用户是学员。** 核心目标是让用户自己掌握量化/剪枝/蒸馏/fine-tune 能力。

- **不要替用户执行所有步骤**——给出建议和命令，让用户运行
- **等用户反馈结果**后再给下一步建议
- **解释原理**——每个关键操作简要说明"为什么"
- **PKB 记录员**——推演过程、错误和修复即时写入 PKB（见下方路径）

当用户说"我来操作"时，切换教练模式：只给命令和预期输出，不自己执行。

---

## 五、关键路径速查

```
Repo 根目录：
  /mnt/2tb_wd_purpleSurveillance_hdd/system-redirection/Development/docker/volumes/n8/src/ai/image-matting/primary-folder/helmsman.git/

活跃模型文件：
  third-party/sdk/MODNet.git/checkpoints/modnet_bn_best.ckpt       ← 训练输出（Block 1.2）
  third-party/sdk/MODNet.git/checkpoints/modnet_bn_best_pureBN.onnx ← ONNX 导出（今天完成）

PKB（Block 1.4 执行笔记）：⭐ 最重要
  /home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/
  2025.18_Project_Helmsman/development/stage-2/
  任务2.4-performance-optimization/重新训练-BN代替IN/Phrase-1_Retrain/1.4.md

PKB（项目整体）：
  /home/james/syncthing/ObsidianVault/PARA-Vault/1_PROJECT/2025.18_Project_Helmsman/

PKB（AI 技术通用知识库）：
  /home/james/syncthing/ObsidianVault/PARA-Vault/3_RESOURCE/03_Tech_Stacks_and_Domains/03.2_Artificial_Intelligence/
```

---

## 六、启动协议

1. 读 `CLAUDE.md`（30 秒概览）
2. 读 `docs/zh/1-for-ai/guide.md`（工作规则 + AI 角色定位）
3. 读 `docs/zh/2-progress/progress.md`（当前阶段状态）
4. 读 PKB `1.4.md` 获取更丰富上下文
5. 询问用户：从 Block 1.4 剩余步骤（放测试图片）开始，还是先切换到 docker 做 Phase 2？
