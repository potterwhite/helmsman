# Helmsman — AI Agent Guide

> **Target audience:** AI coding agents (Claude Code, Cursor, etc.)
> **Read this before touching any code.**
> **English →** [../../en/1-for-ai/guide.md](../../en/1-for-ai/guide.md)

---

## 1. Reading Order (every session)

1. **This file** — understand how to work in this repo
2. **[`codebase_map.md`](codebase_map.md)** — full codebase structure (replaces scanning)
3. **[`../2-progress/progress.md`](../2-progress/progress.md)** — current phase + active tasks
4. **Relevant reference doc** — only if your task requires it (api_spec, domain rules, etc.)

---

## 2. Non-Negotiable Rules

### Code
- All source code and comments must be in **English**
- Communicate with the human in **Chinese (中文)**
- Do **not** end the session — always prompt the user for next steps

### Commits
- **One commit per STEP** — do not accumulate changes and commit at the end
- Follow the commit message format below exactly
- Never commit broken code or failing tests

### Documentation
- After modifying any file listed in `codebase_map.md`, update that file in the same commit
- When a Phase block is completed, update the status in `../2-progress/progress.md`
- When starting new work, add a date group entry to `../2-progress/NEED_TO_DO.md`

---

## 3. Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Type** (required): `feat` · `fix` · `docs` · `refactor` · `perf` · `test` · `build` · `chore`

**Subject** (required): English, ≤70 chars, present tense
- ✅ `feat(rknn): implement zero-copy inference and INT8 quantization pipeline`
- ✅ `fix: handle non-GPU environments in requirements.txt`
- ❌ `Fixed bugs and updated stuff`

**Body** (recommended): bullet points explaining what and why

**Footer** (recommended): `Phase X BlockY.Z complete.`

> **Important**: The project uses **release-please** automation. Commits merged to `main` with
> `feat:` prefix bump minor version; `fix:` bumps patch; `feat!:` (breaking) bumps major.

---

## 4. How to Handle Human Requests

### "Build a new feature"
1. Ask clarifying questions (goal, affected files, breaking changes)
2. Write a plan in `docs/` — **no code yet**
3. Wait for approval
4. Implement step by step, one commit per step

### "There's a bug"
1. Reproduce and understand the root cause
2. Fix it
3. Commit with `fix:` prefix

### "Refactor / optimize something"
1. Write a refactor plan in `docs/architecture/`
2. Wait for approval
3. Execute step by step

---

## 5. Common Pitfalls

| ❌ Wrong | ✅ Right |
|---|---|
| Edit 10 files then do one big commit | Commit after each logical step |
| Start coding without reading codebase_map | Read codebase_map first |
| Edit code, forget to update codebase_map | Always sync codebase_map in same commit |
| Vague commit message "fix bugs" | Specific message naming the root cause |
| Invent new architectural patterns | Follow existing patterns in codebase_map |
| Edit files in `third-party/sdk/MODNet.git/` directly | Edit in `third-party/scripts/modnet/` (they get symlinked in) |
| Hardcode Python version or pip URLs | `func_2_0_setup_env()` in `common.sh` is the single source of truth |
| Normalize input to [-1,1] in C++ frontend | Range is 0–255 float32; RKNN quantization handles normalization internally |
| Use old ONNX Runtime API `GetInputName()` | Use `GetInputNameAllocated()` (updated since v0.5.0) |
| Run `./helmsman build cpp build rk3588s` without `.env` | Create `runtime/cpp/.env` from `.env.example` first |

---

## 6. Key Architecture Facts

- **CLI entry point is `helmsman`** (root Bash script). All sub-scripts in `scripts/` are *sourced*, never executed directly. Never run `scripts/*.sh` standalone.
- **Python target is 3.8.10** — hard-coded in `common.sh` (`PYTHON_Target_VERSION`). All pip pins in `envs/requirements.txt` are calibrated for this version.
- **C++ backend is compile-time selected** — `ENABLE_RKNN_BACKEND` define in CMake. Native (x86) = ONNX Runtime. `rk3588s`/`rv1126bp` presets = RKNN. Never add runtime branching for this.
- **`TensorData` is the pipeline contract** — Frontend populates `orig_width/height` and `pad_top/bottom/left/right`. Backend MUST consume these to crop letterbox and restore original size. Never create `TensorData` without all fields.
- **Input data range is 0–255 float32 in HWC** — Frontend does NOT normalize to [-1,1]. The RKNN quantization pipeline handles this via calibration data.
- **Version number lives in `CHANGELOG.md`** — `arc_extract_version_from_changelog()` in CMake reads it. Never edit version numbers directly in `CMakeLists.txt`.
- **Two working branches**: `main` (releases only, do not work here directly) and `retrain/modnet` (all current work).
- **MODNet submodule is ephemeral** — `./helmsman cleanall` resets it. All permanent changes belong in `third-party/scripts/modnet/`; symlinks are re-created by `./helmsman prepare`.
- **Anti-fusion is critical for RKNN** — Do not introduce `InstanceNormalization` anywhere in the model. The RKNN compiler will detect and reconstruct it, causing CPU fallback and ~40% latency regression.
- **`runtime/cpp/.env` is gitignored and required** — `cpp_build.sh` exits immediately if it doesn't exist. Must be created manually from `.env.example`.

---

## 7. Development Commands

```bash
# Python environment setup & model operations
./helmsman prepare              # Full setup: pyenv + Python 3.8 + venv + deps + MODNet submodule
./helmsman convert              # .ckpt → .onnx (interactive: original or modified variant)
./helmsman inference            # Run Python ONNX inference on an image (interactive)
./helmsman golden               # Generate golden reference binary files (for C++ validation)
./helmsman clean                # Clean build artifacts and build/
./helmsman cleanall             # Nuclear clean: + .venv, models, MODNet submodule reset

# C++ build commands
./helmsman build cpp build              # Incremental build (native, release, shared)
./helmsman build cpp cb                 # Clean + Build + Install (native)
./helmsman build cpp build rk3588s      # Cross-compile for RK3588 (release)
./helmsman build cpp cb rv1126bp debug  # Fresh build for RV1126BP, debug mode
./helmsman build cpp list               # List all available CMake presets
./helmsman build cpp clean rk3588s      # Clean RK3588 build only
./helmsman build cpp test               # Build & run tests (native)

# Direct CMake (bypass helmsman)
cd runtime/cpp
cmake --preset rk3588s-release
cmake --build build/rk3588s-release -j$(nproc)
cmake --install build/rk3588s-release

# Deploy to board
./tools/deploy_and_test.sh              # build rk3588s → rsync → remote infer
./tools/deploy_and_benchmark.sh         # extended benchmark on board

# Validate C++ against Python golden files
python3 tools/MODNet/verify_golden_tensor.py
python3 tools/MODNet/reconstruct_from_bin.py

# C++ binary usage (after install)
# runtime/cpp/install/<platform>/release/bin/Helmsman_Matting_Client
Helmsman_Matting_Client <image_path> <model_path> <output_dir> [background_path]
```
