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

#
# apps/matting/server/src/pipeline/stages/frontend/CMakeLists.txt
#

# Internal stage headers live in src/ (not exported via include/).
# Add src/ to the include search path so #include "pipeline/stages/..." resolves.
get_filename_component(_server_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
target_include_directories(${PROJECT_NAME}
    PRIVATE
        "${_server_src_dir}"
)

# --- Frontend core: base class + shared sources ---
target_sources(${PROJECT_NAME}
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/frontend-core/frontend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/frontend-core/impl/no-hw-frontend.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/stages/04-preprocess/preprocessor.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/stages/01-input-source/ffmpeg-input-source.cpp"
)

# --- Platform-specific: Rockchip hardware decode path ---
# Factory selects RockchipFrontend or NoHwFrontend at compile time.
# See frontend-core/factory/frontend-create-*.cpp for the two
# FrontEnd::Create() implementations.
if("rockchip" IN_LIST CMAKE_PLATFORM)
    target_sources(${PROJECT_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/frontend-core/impl/rockchip-frontend.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/frontend-core/factory/frontend-create-rockchip.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/stages/02-decoder/mpp-frame-decoder.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/stages/03-color-convert/rga-nv12-to-bgr.cpp"
    )
else()
    target_sources(${PROJECT_NAME}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/frontend-core/factory/frontend-create-nohw.cpp"
    )
endif()
