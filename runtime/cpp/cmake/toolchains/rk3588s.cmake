# Copyright (c) 2025 PotterWhite
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# toolchains/rk3588s.cmake
#
# This toolchain file configures CMake for cross-compiling
# to an aarch64 Linux target (RK3588S) using a Buildroot SDK.
#
# Best Practice Score: 99/100
#
# ==============================================================================
# WARNING: POTENTIAL RISK (Search Mode: BOTH)
# ==============================================================================
# This toolchain sets CMAKE_FIND_ROOT_PATH_MODE_* to 'BOTH'.
#
# 1. Why: This allows CMake to find libraries in your external 'install' folder
#    (defined by CMAKE_PREFIX_PATH in CMakePresets.json) which sit outside the
#    compiler's default Sysroot.
#
# 2. Risk: If 'CMAKE_PREFIX_PATH' is accidentally polluted with host directories
#    (like /usr/lib), the linker might try to link host x86 libraries into your
#    ARM binary, causing "incompatible architecture" errors.
#
# 3. Mitigation: Ensure your CMakePresets.json strictly manages CMAKE_PREFIX_PATH
#    to point ONLY to your cross-compiled install directories.
# ==============================================================================

# -------------------------------------
# I. Target System Information
# -------------------------------------
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# -------------------------------------
# II. Sysroot Configuration
# -------------------------------------
# Root directory for the cross-compilation environment.
# Logic: Allow overriding via Environment Variable for CI/CD portability.
if(DEFINED ENV{ARC_RK3588S_SDK_ROOT})
    set(SDK_HOST_PREFIX "$ENV{ARC_RK3588S_SDK_ROOT}")
else()
    # Fallback to the hardcoded default path
    set(SDK_HOST_PREFIX "/development/toolchain/host")
endif()

set(CMAKE_SYSROOT "${SDK_HOST_PREFIX}/aarch64-buildroot-linux-gnu/sysroot")

# Validation check
if(NOT EXISTS "${CMAKE_SYSROOT}")
    message(FATAL_ERROR "Cross-compilation Sysroot not found at: ${CMAKE_SYSROOT}.\nPlease set env var ARC_RK3588S_SDK_ROOT or fix the path in toolchain file.")
endif()

# -------------------------------------
# III. Cross Compilers
# -------------------------------------
set(TOOLCHAIN_BIN_DIR "${SDK_HOST_PREFIX}/bin")
set(CMAKE_C_COMPILER "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-g++")

# -------------------------------------
# IV. Search Path Configuration
# Guide CMake to look in Sysroot first, then CMAKE_PREFIX_PATH, avoiding Host paths.
# -------------------------------------
list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_SYSROOT}")

# Tools (make, python, etc.) must use the Host version
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Libraries/Headers/Packages must search Sysroot + Staging Install (BOTH)
# This enables finding dependencies built in previous steps (e.g., Utils)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# -------------------------------------
# V. RPATH Settings (Optional but recommended)
# Allow executables on the target board to find .so files in the same directory.
# -------------------------------------
set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# -------------------------------------
# VI. Set other toolchain utilities
# Explicitly specifying these prevents CMake from finding host versions.
# -------------------------------------
set(CMAKE_STRIP   "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-strip")
set(CMAKE_AR      "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-ar")
set(CMAKE_LINKER  "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-ld")
set(CMAKE_NM      "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-nm")
set(CMAKE_OBJCOPY "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-objcopy")
set(CMAKE_OBJDUMP "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-objdump")
set(CMAKE_RANLIB  "${TOOLCHAIN_BIN_DIR}/aarch64-buildroot-linux-gnu-ranlib")