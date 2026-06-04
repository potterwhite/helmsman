#!/bin/bash
# Copyright (c) 2026 PotterWhite
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

# ==============================================================================
# Automated Deploy and Benchmark Script
# ==============================================================================
# This script automates the entire workflow:
#   1. Build C++ code for rk3588s
#   2. Rsync binaries and models to target board
#   3. Execute inference with both FP16 and INT8 models
#   4. Compare performance and display results
# ==============================================================================

set -e

# --- Configuration ---
REMOTE_HOST="evboard"
REMOTE_USER="root"
REMOTE_PASSWORD="123"
REMOTE_BASE="/usr"
REMOTE_TEST_DIR="/root"
LOCAL_INSTALL_DIR="runtime/cpp/install/rk3588s/release"
LOCAL_MODELS_DIR="modnet-models"
TEST_IMAGE="${REMOTE_TEST_DIR}/green-fall-girl-point-to.png"
TEST_OUTPUT_DIR="${REMOTE_TEST_DIR}/debug/cpp/"

# Model paths
FP16_MODEL="${REMOTE_TEST_DIR}/modnet_fp16.rknn"
INT8_MODEL="${REMOTE_TEST_DIR}/modnet_int8.rknn"

# Colors
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[Deploy]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[Step]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

log_error() {
    echo -e "${RED}[Error]${NC} $1"
}

log_benchmark() {
    echo -e "${CYAN}[Benchmark]${NC} $1"
}

log_result() {
    echo -e "${MAGENTA}[Result]${NC} $1"
}

# ==============================================================================
# Step 1: Build C++ Code
# ==============================================================================
log_step "Building C++ code for rk3588s..."
./helmsman build cpp build rk3588s

log_step "Installing binaries..."
./helmsman build cpp install rk3588s

# ==============================================================================
# Step 2: Rsync Binaries to Target Board
# ==============================================================================
log_step "Syncing binaries to ${REMOTE_HOST}:${REMOTE_BASE}..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh --progress "${LOCAL_INSTALL_DIR}/" "${REMOTE_HOST}:${REMOTE_BASE}/"

if [ $? -eq 0 ]; then
    log_info "Rsync binaries completed successfully"
else
    log_error "Rsync binaries failed"
    exit 1
fi

# ==============================================================================
# Step 3: Rsync Models to Target Board
# ==============================================================================
log_step "Syncing INT8 model to ${REMOTE_HOST}:${REMOTE_TEST_DIR}..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh --progress "${LOCAL_MODELS_DIR}/modnet_int8.rknn" "${REMOTE_HOST}:${INT8_MODEL}"

if [ $? -eq 0 ]; then
    log_info "INT8 model synced successfully"
else
    log_error "INT8 model sync failed"
    exit 1
fi

# Check if FP16 model exists on remote
log_step "Checking if FP16 model exists on remote..."
FP16_EXISTS=$(sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "[ -f ${FP16_MODEL} ] && echo 'yes' || echo 'no'")

if [ "$FP16_EXISTS" = "no" ]; then
    log_warn "FP16 model not found on remote, will only test INT8"
    TEST_FP16=false
else
    log_info "FP16 model found on remote"
    TEST_FP16=true
fi

# ==============================================================================
# Step 4: Create Output Directory
# ==============================================================================
sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

# ==============================================================================
# Step 5: Benchmark FP16 Model (if exists)
# ==============================================================================
if [ "$TEST_FP16" = true ]; then
    log_benchmark "=========================================="
    log_benchmark "Testing FP16 Model Performance"
    log_benchmark "=========================================="

    log_info "Running: Helmsman_Matting_Server ${TEST_IMAGE} ${FP16_MODEL} ${TEST_OUTPUT_DIR}"

    sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Server ${TEST_IMAGE} ${FP16_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/fp16_inference.log

    FP16_EXIT_CODE=${PIPESTATUS[0]}

    if [ ${FP16_EXIT_CODE} -eq 0 ]; then
        log_info "✅ FP16 inference completed successfully!"

        # Extract performance metrics
        FP16_AVG_TIME=$(grep "Performance Benchmark" /tmp/fp16_inference.log | awk '{print $NF}' | sed 's/ms.//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
        log_result "FP16 Average Inference Time: ${FP16_AVG_TIME} ms"
    else
        log_error "❌ FP16 inference failed with exit code ${FP16_EXIT_CODE}"
    fi

    echo ""
fi

# ==============================================================================
# Step 6: Benchmark INT8 Model
# ==============================================================================
log_benchmark "=========================================="
log_benchmark "Testing INT8 Model Performance"
log_benchmark "=========================================="

log_info "Running: Helmsman_Matting_Server ${TEST_IMAGE} ${INT8_MODEL} ${TEST_OUTPUT_DIR}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Server ${TEST_IMAGE} ${INT8_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_inference.log

INT8_EXIT_CODE=${PIPESTATUS[0]}

if [ ${INT8_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 inference completed successfully!"

    # Extract performance metrics
    INT8_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_inference.log | awk '{print $NF}' | sed 's/ms.//' | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print "N/A"}')
    log_result "INT8 Average Inference Time: ${INT8_AVG_TIME} ms"
else
    log_error "❌ INT8 inference failed with exit code ${INT8_EXIT_CODE}"
    exit 1
fi

echo ""

# ==============================================================================
# Step 7: Performance Comparison
# ==============================================================================
if [ "$TEST_FP16" = true ] && [ ${FP16_EXIT_CODE} -eq 0 ] && [ ${INT8_EXIT_CODE} -eq 0 ]; then
    log_benchmark "=========================================="
    log_benchmark "Performance Comparison Summary"
    log_benchmark "=========================================="

    echo -e "${CYAN}Model Type${NC}     | ${CYAN}Avg Time (ms)${NC} | ${CYAN}FPS${NC}    | ${CYAN}Speedup${NC}"
    echo "-------------- | ------------- | ------ | --------"

    if [ "$FP16_AVG_TIME" != "N/A" ] && [ "$INT8_AVG_TIME" != "N/A" ]; then
        FP16_FPS=$(echo "scale=2; 1000 / $FP16_AVG_TIME" | bc)
        INT8_FPS=$(echo "scale=2; 1000 / $INT8_AVG_TIME" | bc)
        SPEEDUP=$(echo "scale=2; $FP16_AVG_TIME / $INT8_AVG_TIME" | bc)

        printf "%-14s | %-13s | %-6s | %-8s\n" "FP16" "$FP16_AVG_TIME" "$FP16_FPS" "1.00x"
        printf "%-14s | %-13s | %-6s | ${GREEN}%-8s${NC}\n" "INT8" "$INT8_AVG_TIME" "$INT8_FPS" "${SPEEDUP}x"

        echo ""
        log_result "🚀 INT8 quantization achieved ${SPEEDUP}x speedup!"
        log_result "📊 FPS improvement: ${FP16_FPS} → ${INT8_FPS} ($(echo "scale=1; ($INT8_FPS - $FP16_FPS) / $FP16_FPS * 100" | bc)% increase)"
    else
        log_warn "Could not calculate performance metrics"
    fi
else
    log_benchmark "=========================================="
    log_benchmark "INT8 Performance Summary"
    log_benchmark "=========================================="

    if [ "$INT8_AVG_TIME" != "N/A" ]; then
        INT8_FPS=$(echo "scale=2; 1000 / $INT8_AVG_TIME" | bc)
        log_result "INT8 Average Time: ${INT8_AVG_TIME} ms"
        log_result "INT8 FPS: ${INT8_FPS}"
    fi
fi

echo ""

# ==============================================================================
# Step 8: List Output Files
# ==============================================================================
log_step "Output files generated:"
sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "ls -lh ${TEST_OUTPUT_DIR}"

echo ""
log_info "🎉 Benchmark completed successfully!"
log_info "📝 Logs saved to:"
log_info "   - FP16: /tmp/fp16_inference.log"
log_info "   - INT8: /tmp/int8_inference.log"
