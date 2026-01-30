# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.0] - 2025-08-07

### Added

* Reproducible Python 3.8 ONNX conversion environment.
* Initial C++ project scaffold with CMake and aarch64 toolchain support.

### Fixed

* MODNet ONNX export issues related to CPU execution and state_dict mismatches.
* Dependency conflicts in the ONNX conversion pipeline.

### Verified

* End-to-end ONNX conversion and inference for both webcam and photographic models.

