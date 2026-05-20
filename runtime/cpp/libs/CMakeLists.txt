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


# ---------------------------------
# I. Main Executes
# ---------------------------------

# Common libraries (all platforms)
add_subdirectory(utils)
add_subdirectory(network)
add_subdirectory(cvkit)

# Rockchip-specific libraries
if("rockchip" IN_LIST CMAKE_PLATFORM)
    add_subdirectory(mppkit)
    add_subdirectory(rgakit)
endif()

# Linux-specific libraries
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_subdirectory(dmakit)
    add_subdirectory(drmkit)
endif()

# Runtime (all platforms, internal RKNN condition control)
add_subdirectory(runtime)

# ---------------------------------
#               End
# ---------------------------------
