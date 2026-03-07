# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.7.0](https://github.com/potterwhite/helmsman/compare/v0.6.0...v0.7.0) (2026-03-07)


### ⚠ BREAKING CHANGES

* TensorData: added orig_width/height + pad_top/bottom/left/right
* Frontend::preprocess now takes model_width/height (no more hard-coded 512)
* Postprocess: crops padding and resizes back to original resolution


### ✨ Added

* dynamic input shape + letterbox metadata for correct output size ([#12](https://github.com/potterwhite/helmsman/issues/12)) ([035ff06](https://github.com/potterwhite/helmsman/commit/035ff068f33f70a9f8ba3bfc643a2e42053f8a8c))

#### Features:
* Read real model input size from RKNN/ONNX instead of hard-coding
* Propagate letterbox metadata through entire pipeline
* Output image now matches original input dimensions

#### Chore:
* Added benchmark scripts for 384×384, opt levels, normalization variants

---

## [0.6.0](https://github.com/potterwhite/helmsman/compare/v0.5.0...v0.6.0) (2026-03-04)


### ⚠ BREAKING CHANGES

* **rknn:** Refactored InferenceEngine base class and renamed `main-client.h` to `common-define.h`.

### ✨ Added

* **rknn:** implement zero-copy inference and manual INT8 quantization pipeline ([#10](https://github.com/potterwhite/helmsman/issues/10)) ([6566ccc](https://github.com/potterwhite/helmsman/commit/6566cccf42c33daee8adef70a648c3a0e7b9a7c7))

* add deploy_and_test.sh for automated execution and remote testing

* support dual mode: InferenceEngineRKNN (non-zero-copy) and ZeroCP

* inference: reduced latency to 287.66ms through 512x512 optimization

* quantization: bypassed NPU internal normalization overhead by pre-processing in C++

* rename main-client.h to common-define.h for general platform usage

* integrate OtherUtils and improve logging with module-name suffixes

* Refactored InferenceEngine base class and renamed core project headers.

---

## [0.5.0](https://github.com/potterwhite/helmsman/compare/v0.4.0...v0.5.0) (2026-02-12)

* replace Conan-based ONNX Runtime integration with CMake preset-driven FetchContent mechanism ([#8](https://github.com/potterwhite/helmsman/issues/8)) ([9810179](https://github.com/potterwhite/helmsman/commit/9810179785ef58dc8a3ce9a84f2335a0c2c34ed8))

### ⚠ BREAKING CHANGES

- Removed Conan-based ONNX Runtime packaging mechanism
- Removed ONNXRUNTIME_ROOT CMake variable usage
- `MathUtils::getScaleFactor()` now returns `ScaleFactor` struct instead of `std::pair<double,double>`
- Native build no longer depends on Conan for ONNX Runtime integration

### ✨ Added

- CMake preset mixins for ONNX Runtime 1.16.3 (linux-x86_64 / linux-aarch64)
- FetchContent-based ONNX Runtime download and extraction
- Imported CMake target `${PROJECT_NAMESPACE}::ThirdParty::ONNXRuntime`
- Automatic dependency cache support via `ARC_DEP_CACHE_DIR`

### 🛠 Fixed

- Replaced deprecated `GetInputName()` and `GetOutputName()` APIs with `GetInputNameAllocated()`
- Removed manual allocator memory free
- Cleaned preset inheritance structure

---

## [0.4.0](https://github.com/potterwhite/helmsman/compare/v0.3.0...v0.4.0) (2026-02-10)

* **release:** bump version to reflect C++ runtime and script modularization ([#6](https://github.com/potterwhite/helmsman/issues/6)) ([41c5bc4](https://github.com/potterwhite/helmsman/commit/41c5bc4fd928b836a568a1561ae08010ff3286e3))

### ✨ Added
- **Scripts:** created `scripts/` directory containing modularized logic: `common.sh`, `setup.sh`, `python_ops.sh`, `cpp_build.sh`, and `menus.sh`.
- **Build System:** `runtime/cpp/.env.example` for environment variable configuration.
- **Build System:** `runtime/cpp/CMakePresets.json` defining build presets (native, rv1126bp, rk3588s).
- **Build System:** `runtime/cpp/cmake/ArcFunctions.cmake` providing macros for library installation, versioning, and testing.
- **Build System:** Toolchain files in `runtime/cpp/cmake/toolchains/` for cross-compilation.
- **C++ Libs:** `libs/cvkit` for OpenCV image processing wrappers (normalization, resizing).
- **C++ Libs:** `libs/network` containing socket base, client, and server implementations.
- **C++ Libs:** `libs/asr_engine` wrapping Sherpa-Onnx recognizer, VAD, and WAV reading logic.
- **C++ Libs:** `libs/runtime` wrapping ONNX Runtime session management.
- **C++ Apps:** `apps/asr/client`, `apps/asr/server`, and `apps/matting/client` implementations.
- **Testing:** `cmake/FetchGTest.cmake` and `test/` directory structure for unit testing.
- **Docs:** `BUILD_SYSTEM.md` and `ADD_NEW_SOC.md` (English and Chinese) explaining the new architecture.
- **Tools:** `third-party/scripts/modnet/generate_golden_files-v3-instrumentation.py` for generating intermediate debug binaries.

### Changed
- **Shell:** Refactored the root `helmsman` script to act as a wrapper that sources functions from the `scripts/` directory.
- **C++:** Moved C++ source code from a flat `src/` directory to a hierarchical `libs/` and `apps/` structure.
- **Dependencies:** Integrated `librknnrt` and `sherpa-onnx` downloading logic into `runtime/cpp/third_party/` via CMake `FetchContent`.

### Removed
- **Legacy:** Deleted the original monolithic `runtime/cpp/src/main.cpp` and the previous root `CMakeLists.txt`.
- **Legacy:** Removed `archived/helmsman` file.


---

## [0.3.0](https://github.com/potterwhite/helmsman/compare/v0.2.0...v0.3.0) (2026-02-03)


### ✨ Added

* refactor helmsman into full-featured unified CLI + implement bit-exact NumPy preprocessing in C++ ([#3](https://github.com/potterwhite/helmsman/issues/3)) ([7674d50](https://github.com/potterwhite/helmsman/commit/7674d50baa89032409e09143651a62c368129de6))

## [0.2.0](https://github.com/potterwhite/helmsman/compare/v0.1.0...v0.2.0) (2026-01-30)


### ✨ Added

* **release:** add MIT license headers and release-please workflow ([#1](https://github.com/potterwhite/helmsman/issues/1)) ([68c3a65](https://github.com/potterwhite/helmsman/commit/68c3a65b2842b6096adb101512f7598e574546a0))

- MIT license headers added to all source files.
- Release-please workflow added for automated semantic versioning.
- Initial release-please configuration and manifest created.
- Initial CHANGELOG.md created.

### 🐛 Fixed
- Update MODNet submodule to ZHKKKe's official repository.


---

## [0.1.0] - 2025-08-07

### Added

* Reproducible Python 3.8 ONNX conversion environment.
* Initial C++ project scaffold with CMake and aarch64 toolchain support.

### Fixed

* MODNet ONNX export issues related to CPU execution and state_dict mismatches.
* Dependency conflicts in the ONNX conversion pipeline.

### Verified

* End-to-end ONNX conversion and inference for both webcam and photographic models.
