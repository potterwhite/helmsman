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
# apps/matting/server/src/CMakeLists.txt
#

#################################
# 1.0 input/
set(INPUT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/input/mp4-input-source.cpp"
)

#################################
# 2.0 pipeline/

# 2.1 pipeline/stages/frontend/
set(PIPELINE_FRONTEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/frontend/frontend.cpp"
)

# 2.2 pipeline/stages/inference-engine/
set(PIPELINE_INFERENCE_ENGINE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/inference-engine/onnx/onnx.cpp"
)

if(TARGET ${PROJECT_NAMESPACE}::ThirdParty::LibRKNNRT)
    list(APPEND PIPELINE_INFERENCE_ENGINE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/inference-engine/rknn/rknn-zero-copy.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/inference-engine/rknn/rknn-non-zero-copy.cpp"
    )
endif()

# 2.3 pipeline/stages/backend
set(PIPELINE_BACKEND_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/backend/backend.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/stages/backend/post-processor/guided-filter-post-processor.cpp"
)

# 2.4 pipeline/
set(PIPELINE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/pipeline.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/recurrent-state-manager.cpp"
    ${PIPELINE_FRONTEND_SOURCES}
    ${PIPELINE_INFERENCE_ENGINE_SOURCES}
    ${PIPELINE_BACKEND_SOURCES}
)

#################################
# 3.0 ./
set(SERVER_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/main-server.cpp"
)

target_sources(${PROJECT_NAME}
    PRIVATE
        ${INPUT_SOURCES}
        ${PIPELINE_SOURCES}
        ${SERVER_SOURCE}
)
