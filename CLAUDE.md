# CLAUDE.md

This file provides essential guidance to Claude Code when working in this repository.
**For full context, always start by reading the docs** (see below).

## ⚠️ Session Start Protocol

**Before writing any code**, read these two files in order:
1. 从`docs/zh/2-progress/NEED_TO_DO.md`读取当前的紧急任务
2. `docs/zh/1-for-ai/guide.md` — working rules, commit format, key architecture facts
3. `docs/zh/1-for-ai/codebase_map.md` — full codebase map (do NOT scan files instead)

Then check `docs/zh/2-progress/progress.md` for current phase status.


Do **not** scan `runtime/cpp/`, `third-party/`, or `scripts/` before reading the above.

---

## Requirements

- All source code and comments must be in **English**
- Communicate with me in **Chinese (中文)**
- Do not end the session arbitrarily — always wait on the UI of the tool of UserAskQuestion

---

## Commands

```bash
./helmsman prepare              # Full setup: pyenv + venv + deps + MODNet submodule
./helmsman convert              # .ckpt → .onnx (interactive)
./helmsman inference            # Python ONNX inference on an image
./helmsman golden               # Generate golden reference binary files
./helmsman build cpp build      # Incremental C++ build (native, release)
./helmsman build cpp cb rk3588s # Clean + build + install for RK3588
./helmsman build cpp list       # List all CMake presets
./helmsman clean                # Clean build artifacts
./helmsman cleanall             # Clean everything including .venv and models
```

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
