# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.10.0](https://github.com/potterwhite/helmsman/compare/v0.9.0...v0.10.0) (2026-06-04)


### ✨ Added

* add --no-prefetch flag to disable prefetch worker thread ([71672fb](https://github.com/potterwhite/helmsman/commit/71672fbf51ed9bf9205ef6b1fb66f86878d42caa))
* add RVM quality investigation script (Block 5.Q) ([2fefdc2](https://github.com/potterwhite/helmsman/commit/2fefdc2f0f4ba315fd565472fff1463bceb621fe))
* **backend:** add no-resize RVM guided filter postprocess with normalization fix ([2a0629c](https://github.com/potterwhite/helmsman/commit/2a0629cd2f88177c3f8ce87bf186d947e8683dec))
* **cli:** add --rvm / --modnet flags for model type selection ([e16afc9](https://github.com/potterwhite/helmsman/commit/e16afc9d73390a207530bde4bd9c2befcab5d885))
* **diag:** add --diag-enabled flag for internal state inspection ([00cb241](https://github.com/potterwhite/helmsman/commit/00cb24115fbf8c41c40cc861c71f5702f0a189e5))
* **dmakit,server:** DMA zero-copy output pipeline (Block 5.8-s6) ([c64e4fb](https://github.com/potterwhite/helmsman/commit/c64e4fb466cf13b577d9a9109a37ba8ac41d0ba5))
* **drmkit:** minimal legacy KMS dumb-buffer display sink (s11) ([c3a8a2a](https://github.com/potterwhite/helmsman/commit/c3a8a2a9e28055c1cf4c879f67894d80705fd5c7))
* **engine:** add GetLastSubTimings() for per-frame sub-step reporting ([719ec6f](https://github.com/potterwhite/helmsman/commit/719ec6fdb5b9771b58a50f6b1a49b1d1ed7009e5))
* **engine:** RKNN ZeroCopy DoInfer sub-step timing ([4c96c50](https://github.com/potterwhite/helmsman/commit/4c96c50cd3b211c03ac5141b385fad35c83f6d90))
* **helmsman:** replace positional args with --key=value flags for build cpp ([bb13958](https://github.com/potterwhite/helmsman/commit/bb13958815066f4f8cf4a69a075187c457c26e25))
* **mppkit:** add MPPKit library — RK3588 MPP encoder/decoder wrapper (Block 5.8-s5.1) ([9fcac87](https://github.com/potterwhite/helmsman/commit/9fcac870e26d547d3e1ef568d0d5f295d90c9ff4))
* **phase-5:** Block 5.1 — MP4 video input pipeline with InputSource abstraction ([ae3abc1](https://github.com/potterwhite/helmsman/commit/ae3abc101deb05ac1fe4974c1c5228304378630b))
* **phase-5:** Block 5.1.5 — video compositing output + SIGINT graceful stop ([055d402](https://github.com/potterwhite/helmsman/commit/055d4020991e36f7f9c9e58ec583812ed0dccf0a))
* **phase-5:** Block 5.2 — dual-buffer prefetch pipeline ([cdc6ec2](https://github.com/potterwhite/helmsman/commit/cdc6ec2b19d640ca62fb42af51ab9a7001b8289d))
* refactor pipeline ([879ea45](https://github.com/potterwhite/helmsman/commit/879ea4536489b5ae7fad3bd8e3bd0a50c9323ea9))
* **rgakit:** add RGAKit library — RK3588 RGA 2D graphics wrapper (Block 5.8-s5.2) ([685a857](https://github.com/potterwhite/helmsman/commit/685a857ff5f3f7763f09ee0beb13c2c335732c80))
* **rknnkit:** add RKNNMemory zero-copy allocator with selective non-cacheable ([8776fba](https://github.com/potterwhite/helmsman/commit/8776fbaf0cc5aa8dd2c3d495450054ea0d4ba452))
* **rknnkit:** extract RKNN query utilities into independent library ([0690a9e](https://github.com/potterwhite/helmsman/commit/0690a9e262ac34dae894610a4ae0d41d1f36135e))
* **rvm-benchmark:** add --H/--W args for configurable primary resolution ([4b3f83d](https://github.com/potterwhite/helmsman/commit/4b3f83da03553ddebc55ae64e98c68a5636b26d9))
* **rvm:** add --output=mp4|drm flag with DRM display path (s12) ([b9a4130](https://github.com/potterwhite/helmsman/commit/b9a41301e4e856c2d887cc718c8a9bd8a140adf9))
* **rvm:** add A1 DMA buffer swap infrastructure (disabled by default) ([57eb18d](https://github.com/potterwhite/helmsman/commit/57eb18d4168fb489db3b3ea2714d676f8c6522cd))
* **rvm:** add full pipeline timing coverage (s10) ([9f82f64](https://github.com/potterwhite/helmsman/commit/9f82f64f75900abdd957faa68fa940075552e9a1))
* **rvm:** add RKNN resize path in frontend + helmsman compare tool ([081f438](https://github.com/potterwhite/helmsman/commit/081f438e8bfbd9773490812653e3622b8ffb45c2))
* **rvm:** add RobustVideoMatting as git submodule ([e89e709](https://github.com/potterwhite/helmsman/commit/e89e709b0a023da7436cd2327f6a9e8a9db32bdc))
* **rvm:** add rvm-models dir, onnx-simplifier dep, and rvm-models gitignore ([b1cbdfe](https://github.com/potterwhite/helmsman/commit/b1cbdfee05728cb82f3a3ade57a5713291f9ecc0))
* **rvm:** auto-detect recurrent state shapes from RKNN model (For Block5.8-s7) ([600af2b](https://github.com/potterwhite/helmsman/commit/600af2b12c5fbbd4ceddb837a44a0d0a5e45da14))
* **rvm:** complete Phase-3 C++ integration — multi-tensor pipeline + RecurrentStateManager ([767a492](https://github.com/potterwhite/helmsman/commit/767a492838ff446f74df529e266db4644d619589))
* **rvm:** wire GuidedFilter post-processor into video inference loop ([3f3d0a0](https://github.com/potterwhite/helmsman/commit/3f3d0a04e37a78cc5c7a537f921d8f4aa0b04ac8))
* **s18:** Frontend pure-separation architecture — MPP decode ready ([f584be8](https://github.com/potterwhite/helmsman/commit/f584be8dc1bbbee4805424486521fdfbd67d211d))
* **s18:** wire up MPP hardware decode path with --mpp flag ([65255d4](https://github.com/potterwhite/helmsman/commit/65255d4e95c245e3320e4625056717a08637bc95))
* **s22:** add --perf-enabled CLI flag + inference/backend scope comments ([2e9475a](https://github.com/potterwhite/helmsman/commit/2e9475a7e4c9fedb817d34632a8501dc489e6bde))
* **s22:** add RKNN perf_detail/profiling instrumentation + --core-mask CLI (s22-1) ([7c9be1f](https://github.com/potterwhite/helmsman/commit/7c9be1f23d3d82d4d0c584221a98af939066632f))
* **timer:** add preprocess::resize sub-timer for s5_8_22_9 baseline ([fa10ed5](https://github.com/potterwhite/helmsman/commit/fa10ed5fd2479df0735ce0377bbefe41b54432ca))
* **timer:** format ScopedTimer output with human-friendly units ([ba05685](https://github.com/potterwhite/helmsman/commit/ba0568500737e61025e3e13e8f251c8b65eec4ef))
* **timing:** add group headers, simplified names, and per-frame timing output ([04c394f](https://github.com/potterwhite/helmsman/commit/04c394f698eba300277fa2bf808fce731f7f84be))
* **version:** add git version info auto-injection with --info CLI command ([c420b72](https://github.com/potterwhite/helmsman/commit/c420b72076b72b38df0d5dc7d5bc50743a724ade))


### 🐛 Fixed

* --backend=onnx ([4f23bda](https://github.com/potterwhite/helmsman/commit/4f23bda3ab526e5b2ef79225ce8a39532a004e72))
* add missing &lt;functional&gt; include and gate no-resize log behind diag_enabled ([19788bd](https://github.com/potterwhite/helmsman/commit/19788bd1ea2a0f00dde5394582c726c0cce1d9b6))
* **backend:** gate no-resize debug data log behind --inspect (diag_enabled_) ([77ba2bf](https://github.com/potterwhite/helmsman/commit/77ba2bf9a63f11a58bed3399e7e0368eb883c5a2))
* **backend:** use full type name in timing accessors ([57039c2](https://github.com/potterwhite/helmsman/commit/57039c2f4f9abbde97da39261bff427571a45249))
* CMake CMP0135 policy warnings for third party dependencies ([fcbe2a3](https://github.com/potterwhite/helmsman/commit/fcbe2a3232e4046b03b7716b1e740fe66da192fd))
* **cmake:** use real UTC time in build timestamp ([190f279](https://github.com/potterwhite/helmsman/commit/190f279a52de50864645ccde823a24e5a6ee48b8))
* constexpr kDownsampleRatio moved to global field of pipeline.cpp ([d437f1d](https://github.com/potterwhite/helmsman/commit/d437f1d44126dc11bbe43261f5f361469c7c516b))
* **ffmpeg:** merge BSF outputs and flush at EOF ([9bcd2bd](https://github.com/potterwhite/helmsman/commit/9bcd2bd67a83ba385d78ec45fd273c1c93c017e2))
* **logging:** move frame header before inference and skip FPS at frame 1 ([f756ea8](https://github.com/potterwhite/helmsman/commit/f756ea8ac9500e780b328caa7c71ba52fca9c155))
* MPPKit header file not found error ([1e006d8](https://github.com/potterwhite/helmsman/commit/1e006d818d8b63af78ccd6db2d9e282f370efb27))
* **mpp:** use h264_mp4toannexb BSF for AVCC→Annex B conversion ([2a01a13](https://github.com/potterwhite/helmsman/commit/2a01a13c4f601fadcdd8d593bcd8c8fbc3b11e63))
* remove double config_ prefix in ScopedTimer call ([390e1ae](https://github.com/potterwhite/helmsman/commit/390e1aee55162982b51cd7d2af88975e1bd53322))
* **rga:** allow DMA buffer fd in Execute null-checks ([b0c4567](https://github.com/potterwhite/helmsman/commit/b0c456763da8b58905076f66432f0010ac615a74))
* **rgakit:** correct wrapbuffer_virtualaddr macro parameter order ([ca2dc5a](https://github.com/potterwhite/helmsman/commit/ca2dc5a890b0d78aa56fbbc5c23b2342bb376a6a))
* roll back normalization modification in ([d5001f5](https://github.com/potterwhite/helmsman/commit/d5001f52d85938ca0b1c4bb588e26598e99205f5))
* rvm composite background picture will fail and warn ([0dfc58c](https://github.com/potterwhite/helmsman/commit/0dfc58ccd9f07edd9eb79a8db14a891a7ef5d62c))
* **rvm-benchmark:** correct input layout from NCHW to NHWC and fix SDK version decoding ([e5744ef](https://github.com/potterwhite/helmsman/commit/e5744efdf004061099503593db47fbd74f91b0ea))
* **rvm:** align default fallback background colour to e.py baseline ([b03fd16](https://github.com/potterwhite/helmsman/commit/b03fd16dd891ab3f74f803113b84ac6646041ea7))
* **rvm:** comment out unused DMA output path to fix build ([feae922](https://github.com/potterwhite/helmsman/commit/feae922883c99c40d5b855a016e1d99bc260cc36))
* **rvm:** correct end-of-frame log off-by-one in frame numbering ([3215cd3](https://github.com/potterwhite/helmsman/commit/3215cd36cece60295a2696f414a4c4c1610b948f))
* **rvm:** correct state shapes and ONNX input for RVM pipeline ([6621691](https://github.com/potterwhite/helmsman/commit/6621691507809391f8bc8ea29b4f29f041f14e5e))
* **rvm:** disable DMA output to restore video file export for quality experiments ([5dccffc](https://github.com/potterwhite/helmsman/commit/5dccffceacb1a9f4e51b5abfc9e024c3a21b1126))
* **rvm:** remove RGA composite, use CPU blend for alpha compositing ([ab0e556](https://github.com/potterwhite/helmsman/commit/ab0e556195292b3560f835e07f00762a25743deb))
* **rvm:** replace letterbox-resize with replicate-pad + dynamic dsr ([da87a8e](https://github.com/potterwhite/helmsman/commit/da87a8e365247ac4bf0babeb543120fbc363e7be))
* **s22:** retry rknn_init without COLLECT_PERF_MASK on unsupported SDKs ([5be1ac5](https://github.com/potterwhite/helmsman/commit/5be1ac5627cf243c953f074d782b769e1a9de253))
* update include paths for stages directory structure ([a7d4f9a](https://github.com/potterwhite/helmsman/commit/a7d4f9ac38290dbdbb4802a1d6a239d3bcdda1b0))
* use PkgConfig imported targets for FFmpeg linking ([7890efe](https://github.com/potterwhite/helmsman/commit/7890efe6f2ae04a39cca0813f190f69a97768b27))


### ⚡ Improved

* add HELMSMAN_DUMP env switch to disable binary dumps at runtime ([8c5d905](https://github.com/potterwhite/helmsman/commit/8c5d905f2fd76dff0c80c4a3ed900eed99d10295))
* replace per-frame std::async with persistent prefetch worker thread ([31d0413](https://github.com/potterwhite/helmsman/commit/31d0413a4adee00898fa8d9753fb66e44458aa19))
* reuse CVKit Base instance across frames in ImageFrontend ([bf59943](https://github.com/potterwhite/helmsman/commit/bf59943a08872f64b09bc1736bff2cdbc1857bed))
* **rknn:** skip fgr output D→H transfer for 7.3% faster inference ([e2dff7c](https://github.com/potterwhite/helmsman/commit/e2dff7c2f9231dfd03f127f0d2424e5a8e440871))
* **rvm:** add per-sub-operation timing in compositeAndWrite (Block 5.8-s4) ([96846a1](https://github.com/potterwhite/helmsman/commit/96846a188982350440c5efb556d2ab5a4cc21edd))
* **rvm:** composite at model resolution, 145ms→95ms (Block 5.8) ([2bd1dbc](https://github.com/potterwhite/helmsman/commit/2bd1dbceee19999a18eecbfe3718fa284b406749))
* **rvm:** replace CPU composite with RGA hardware composite (Block 5.8-s5.3) ([95f74e2](https://github.com/potterwhite/helmsman/commit/95f74e2ec99fd58c7958c1eecb6eb1cd449b11cc))
* **rvm:** separate infer/composite timing + per-frame log (Block 5.8-s4b) ([f14d620](https://github.com/potterwhite/helmsman/commit/f14d620645a205ac7e25da613784069def090b67))
* **rvm:** skip RGA for alpha resize, use CPU directly ([ba19b08](https://github.com/potterwhite/helmsman/commit/ba19b08f132bb98cdf06398c719d58d714cde5df))
* **rvm:** uint8 compositing, replace float32 blend (Block 5.8-s3) ([8c87517](https://github.com/potterwhite/helmsman/commit/8c87517c843cee5e413c414567bc1bd47b3a370e))
* three CPU-side hot-path optimizations in frontend and backend ([b9e514c](https://github.com/potterwhite/helmsman/commit/b9e514cd02b120f9d2fce9e312d1f8b82f52ea30))


### 🔙 Reverts

* **rvm:** remove [DIAG] state_mgr_.reset() — hypothesis 1 disproven ([c352ebe](https://github.com/potterwhite/helmsman/commit/c352ebe7451e1918910f28e8a0c424b7a8d75215))

## [0.9.0](https://github.com/potterwhite/helmsman/compare/v0.8.0...v0.9.0) (2026-04-08)


### ✨ Added

* Pure-BN MODNet end-to-end pipeline — retrain, ONNX export, C++ GuidedFilter, RKNN prep ([#16](https://github.com/potterwhite/helmsman/issues/16)) ([bee7cf3](https://github.com/potterwhite/helmsman/commit/bee7cf3f098783d6311e786ac2f63a5061eb1c01))

## [0.8.0](https://github.com/potterwhite/helmsman/compare/v0.7.0...v0.8.0) (2026-03-12)


### ✨ Added

* **modnet:** eliminate InstanceNorm via graph rewrite to avoid NPU fallback  ([#14](https://github.com/potterwhite/helmsman/issues/14)) ([d804b01](https://github.com/potterwhite/helmsman/commit/d804b01043ca6d347a77fede6a604394b406cc9c))

- Rewrite IBNorm InstanceNormalization branch using mathematically equivalent primitive ops (mean/mul/sqrt) based on Var(x) = E[x²] − (E[x])².

- This intentionally breaks RKNN fusion patterns ("anti-fusion") so the compiler cannot reconstruct InstanceNormalization, preventing CPU transpose fallback and eliminating cross-device memory copies.

- Result: ~40% latency improvement at 576x1024 on RK3588 (~430ms with all NPU cores).

- Note: model architecture modifications live in Helmsman.git. ArcFoundry remains a generic ONNX → RKNN conversion framework.


---

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
