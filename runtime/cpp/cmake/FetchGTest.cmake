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

# ------------------------------------------------------------------------------
# Module: FetchGTest
# Description:
#   Downloads and integrates GoogleTest framework using CMake's FetchContent.
#   This module includes a "Fail-Fast" mechanism to ensure the build stops
#   immediately if the network request fails or the target is invalid.
# ------------------------------------------------------------------------------

include(FetchContent)

# ---------------------------------
# 1. Configuration
# ---------------------------------
# Prevent GoogleTest from overriding our parent project's compiler/linker settings
# on Windows (forces shared CRT).
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# ---------------------------------
# 2. Declare Dependency
# ---------------------------------
message(STATUS "[Testing] Preparing to fetch GoogleTest (v1.14.0)...")

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
    # Recommended hash for v1.14.0 to ensure integrity
    URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
)

# ---------------------------------
# 3. Make Available (Download & Configure)
# ---------------------------------
# This step attempts to download, unpack, and add the sub-directory.
FetchContent_MakeAvailable(googletest)

# ---------------------------------
# 4. Fail-Fast Check (Crucial for Network Issues)
# ---------------------------------
# If FetchContent failed due to network issues, it might not throw a fatal error
# immediately in some older CMake versions, or it might leave a phantom state.
# We explicitly check if the 'gtest' target was successfully created.
# ---------------------------------
if(NOT TARGET gtest)
    message(FATAL_ERROR
        "\n[Testing] Error: GoogleTest target 'gtest' was NOT found after FetchContent.\n"
        "Possible causes:\n"
        "  1. Network connection failure (GitHub is unreachable).\n"
        "  2. Proxy misconfiguration.\n"
        "  3. Corrupted download cache.\n\n"
        "Action required:\n"
        "  - Check your internet connection.\n"
        "  - Try clearing the build directory and re-running cmake.\n"
    )
endif()

# ---------------------------------
# 5. Environment Setup
# ---------------------------------
include(GoogleTest)
message(STATUS "[Testing] GoogleTest v1.14.0 integrated successfully.")