# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


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
