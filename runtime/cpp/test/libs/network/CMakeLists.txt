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

# test/libs/network/CMakeLists.txt

# ---------------------------------
# I. Protection for standalone Use
# ---------------------------------
if(NOT DEFINED GLOBAL_VERSION_STRING OR "${GLOBAL_VERSION_STRING}" STREQUAL "")
    set(GLOBAL_VERSION_STRING "99.99.99")
    message(WARNING "Expected Version is missing, Using Default Version: ${GLOBAL_VERSION_STRING}")
endif()

# ---------------------------------
# II. Project Name
# ---------------------------------
set(PROJECT_NAME "Test_Network")
project(${PROJECT_NAME}
    VERSION
        ${GLOBAL_VERSION_STRING}
    LANGUAGES
        CXX
)

# ---------------------------------
# III. Register Test Target
#    Uses the 'arc_add_test' macro to automate:
#      1. Creating executable 'test_Network'
#      2. Linking GTest & GMock
#      3. Linking the target module (ArcForge::Network)
#      4. Injecting internal 'src' include paths
# ---------------------------------

set(BE_TEST_MODULE "Network")

# Note: The first argument 'Network' must match the library target name defined in libs/network
arc_add_test(${BE_TEST_MODULE}
    test_network.cpp
)

