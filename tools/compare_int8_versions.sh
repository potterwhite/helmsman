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

# Compare INT8 v1 (with normalization) vs v2 (without normalization)

set -e

REMOTE_HOST="evboard"
REMOTE_PASSWORD="123"
REMOTE_TEST_DIR="/root"
LOCAL_MODELS_DIR="modnet-models"
TEST_IMAGE="${REMOTE_TEST_DIR}/green-fall-girl-point-to.png"
TEST_OUTPUT_DIR="${REMOTE_TEST_DIR}/debug/cpp/"

INT8_V1_MODEL="${REMOTE_TEST_DIR}/modnet_int8_v1.rknn"
INT8_V2_MODEL="${REMOTE_TEST_DIR}/modnet_int8_v2.rknn"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[Deploy]${NC} $1"; }
log_step() { echo -e "${BLUE}[Step]${NC} $1"; }
log_benchmark() { echo -e "${CYAN}[Benchmark]${NC} $1"; }
log_result() { echo -e "${MAGENTA}[Result]${NC} $1"; }

# ==============================================================================
# Step 1: Sync both models
# ==============================================================================
log_step "Syncing INT8 v1 (with normalization) to remote..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh "${LOCAL_MODELS_DIR}/modnet_int8.rknn" "${REMOTE_HOST}:${INT8_V1_MODEL}"

log_step "Syncing INT8 v2 (without normalization) to remote..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh "${LOCAL_MODELS_DIR}/modnet_int8_v2.rknn" "${REMOTE_HOST}:${INT8_V2_MODEL}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

# ==============================================================================
# Step 2: Benchmark INT8 v1 (with normalization)
# ==============================================================================
log_benchmark "=========================================="
log_benchmark "Testing INT8 v1 (WITH normalization)"
log_benchmark "=========================================="

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Server ${TEST_IMAGE} ${INT8_V1_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_v1_inference.log

V1_EXIT_CODE=${PIPESTATUS[0]}

if [ ${V1_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 v1 inference completed"
    V1_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_v1_inference.log | grep -oP '\d+\.\d+(?= ms)' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    log_result "INT8 v1 Average Time: ${V1_AVG_TIME} ms"
else
    log_error "❌ INT8 v1 inference failed"
    exit 1
fi

echo ""

# ==============================================================================
# Step 3: Benchmark INT8 v2 (without normalization)
# ==============================================================================
log_benchmark "=========================================="
log_benchmark "Testing INT8 v2 (WITHOUT normalization)"
log_benchmark "=========================================="

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Server ${TEST_IMAGE} ${INT8_V2_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_v2_inference.log

V2_EXIT_CODE=${PIPESTATUS[0]}

if [ ${V2_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 v2 inference completed"
    V2_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_v2_inference.log | grep -oP '\d+\.\d+(?= ms)' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    log_result "INT8 v2 Average Time: ${V2_AVG_TIME} ms"
else
    log_error "❌ INT8 v2 inference failed"
    exit 1
fi

echo ""

# ==============================================================================
# Step 4: Performance Comparison
# ==============================================================================
log_benchmark "=========================================="
log_benchmark "Performance Comparison"
log_benchmark "=========================================="

echo -e "${CYAN}Version${NC}        | ${CYAN}Avg Time (ms)${NC} | ${CYAN}FPS${NC}    | ${CYAN}vs FP16 Baseline${NC}"
echo "-------------- | ------------- | ------ | ----------------"

FP16_BASELINE=287
FP16_FPS=$(echo "scale=2; 1000 / $FP16_BASELINE" | bc)

if [ "$V1_AVG_TIME" != "N/A" ]; then
    V1_FPS=$(echo "scale=2; 1000 / $V1_AVG_TIME" | bc)
    V1_SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $V1_AVG_TIME" | bc)
    printf "%-14s | %-13s | %-6s | %-16s\n" "FP16 Baseline" "$FP16_BASELINE" "$FP16_FPS" "1.00x"
    printf "%-14s | %-13s | %-6s | %-16s\n" "INT8 v1 (norm)" "$V1_AVG_TIME" "$V1_FPS" "${V1_SPEEDUP}x"
fi

if [ "$V2_AVG_TIME" != "N/A" ]; then
    V2_FPS=$(echo "scale=2; 1000 / $V2_AVG_TIME" | bc)
    V2_SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $V2_AVG_TIME" | bc)
    printf "%-14s | %-13s | %-6s | ${GREEN}%-16s${NC}\n" "INT8 v2 (no)" "$V2_AVG_TIME" "$V2_FPS" "${V2_SPEEDUP}x"
fi

echo ""

if [ "$V1_AVG_TIME" != "N/A" ] && [ "$V2_AVG_TIME" != "N/A" ]; then
    IMPROVEMENT=$(echo "scale=2; ($V1_AVG_TIME - $V2_AVG_TIME) / $V1_AVG_TIME * 100" | bc)
    log_result "🚀 Removing normalization improved performance by ${IMPROVEMENT}%"
    log_result "📊 INT8 v2 is $(echo "scale=2; $V1_AVG_TIME / $V2_AVG_TIME" | bc)x faster than INT8 v1"
fi

echo ""
log_info "🎉 Comparison completed!"
log_info "📝 Logs saved to:"
log_info "   - INT8 v1: /tmp/int8_v1_inference.log"
log_info "   - INT8 v2: /tmp/int8_v2_inference.log"
