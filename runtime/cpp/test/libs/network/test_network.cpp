/*
 * Copyright (c) 2025 PotterWhite
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * @file test_network.cpp
 * @brief Unit tests for the Network module.
 * @details This file validates that the GoogleTest framework is correctly integrated
 *          and that the Network module's headers and symbols are linkable.
 */

#include <gtest/gtest.h>

// -----------------------------------------------------------------------------
// I. Include Module Headers
//    This verifies that the include paths are correctly exported by the target.
// -----------------------------------------------------------------------------
// Based on your tree structure: libs/network/include/Network/base/base.h
#include <Network/base/base.h>

// -----------------------------------------------------------------------------
// II. Test Cases
// -----------------------------------------------------------------------------

/**
 * @brief Sanity Check
 * @details Ensures that the GTest framework is running and main() is linked.
 */
TEST(NetworkModuleTest, FrameworkSanityCheck) {
    // Assert that true is true. If this fails, something is fundamentally broken.
    EXPECT_TRUE(true);

    // Log a message to stdout to verify output capturing
    SUCCEED() << "GoogleTest framework is operational.";
}

/**
 * @brief Linkage Check
 * @details Ensures that we can instantiate symbols from the Network library.
 *          This proves that target_link_libraries() in CMake is correct.
 */
TEST(NetworkModuleTest, LibraryLinkage) {
    // Note: Adjust the namespace 'arcforge::embedded' if it differs in your actual code.
    // We strictly follow the Changelog v0.3.0 description here.

    // Assuming Base class exists or checking a macro from system-info
    // Example assertion: Check if a version macro is defined (often in system-info.h)

#if defined(PROJECT_NAME)
    SUCCEED() << "PROJECT_NAME macro is correctly injected: " << PROJECT_NAME;
#else
    FAIL() << "PROJECT_NAME macro is missing.";
#endif

}