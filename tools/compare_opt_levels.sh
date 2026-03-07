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

# Compare INT8 optimization_level 2 vs 3

set -e

REMOTE_HOST="evboard"
REMOTE_PASSWORD="123"
REMOTE_TEST_DIR="/root"
LOCAL_MODELS_DIR="modnet-models"
TEST_IMAGE="${REMOTE_TEST_DIR}/green-fall-girl-point-to.png"
TEST_OUTPUT_DIR="${REMOTE_TEST_DIR}/debug/cpp/"

INT8_OPT2_MODEL="${REMOTE_TEST_DIR}/modnet_int8_opt2.rknn"
INT8_OPT3_MODEL="${REMOTE_TEST_DIR}/modnet_int8_opt3.rknn"

GREEN='\033[1;32m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
MAGENTA='\033[1;35m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[Deploy]${NC} $1"; }
log_step() { echo -e "${BLUE}[Step]${NC} $1"; }
log_benchmark() { echo -e "${CYAN}[Benchmark]${NC} $1"; }
log_result() { echo -e "${MAGENTA}[Result]${NC} $1"; }

# Sync models
log_step "Syncing INT8 opt2 model..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh "${LOCAL_MODELS_DIR}/modnet_int8_v2.rknn" "${REMOTE_HOST}:${INT8_OPT2_MODEL}"

log_step "Syncing INT8 opt3 model..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh "${LOCAL_MODELS_DIR}/modnet_int8_opt3.rknn" "${REMOTE_HOST}:${INT8_OPT3_MODEL}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

# Test opt2
log_benchmark "=========================================="
log_benchmark "Testing INT8 optimization_level=2"
log_benchmark "=========================================="

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Client ${TEST_IMAGE} ${INT8_OPT2_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_opt2_inference.log

OPT2_EXIT_CODE=${PIPESTATUS[0]}

if [ ${OPT2_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 opt2 completed"
    OPT2_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_opt2_inference.log | grep -oP '\d+\.\d+(?= ms)' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    log_result "INT8 opt2 Average Time: ${OPT2_AVG_TIME} ms"
else
    log_error "❌ INT8 opt2 failed"
    exit 1
fi

echo ""

# Test opt3
log_benchmark "=========================================="
log_benchmark "Testing INT8 optimization_level=3"
log_benchmark "=========================================="

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Client ${TEST_IMAGE} ${INT8_OPT3_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_opt3_inference.log

OPT3_EXIT_CODE=${PIPESTATUS[0]}

if [ ${OPT3_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 opt3 completed"
    OPT3_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_opt3_inference.log | grep -oP '\d+\.\d+(?= ms)' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    log_result "INT8 opt3 Average Time: ${OPT3_AVG_TIME} ms"
else
    log_error "❌ INT8 opt3 failed"
    exit 1
fi

echo ""

# Comparison
log_benchmark "=========================================="
log_benchmark "Performance Comparison"
log_benchmark "=========================================="

echo -e "${CYAN}Version${NC}        | ${CYAN}Avg Time (ms)${NC} | ${CYAN}FPS${NC}    | ${CYAN}vs FP16${NC}"
echo "-------------- | ------------- | ------ | --------"

FP16_BASELINE=287
FP16_FPS=$(echo "scale=2; 1000 / $FP16_BASELINE" | bc)

printf "%-14s | %-13s | %-6s | %-8s\n" "FP16 Baseline" "$FP16_BASELINE" "$FP16_FPS" "1.00x"

if [ "$OPT2_AVG_TIME" != "N/A" ]; then
    OPT2_FPS=$(echo "scale=2; 1000 / $OPT2_AVG_TIME" | bc)
    OPT2_SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $OPT2_AVG_TIME" | bc)
    printf "%-14s | %-13s | %-6s | %-8s\n" "INT8 opt2" "$OPT2_AVG_TIME" "$OPT2_FPS" "${OPT2_SPEEDUP}x"
fi

if [ "$OPT3_AVG_TIME" != "N/A" ]; then
    OPT3_FPS=$(echo "scale=2; 1000 / $OPT3_AVG_TIME" | bc)
    OPT3_SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $OPT3_AVG_TIME" | bc)
    printf "%-14s | %-13s | %-6s | ${GREEN}%-8s${NC}\n" "INT8 opt3" "$OPT3_AVG_TIME" "$OPT3_FPS" "${OPT3_SPEEDUP}x"
fi

echo ""

if [ "$OPT2_AVG_TIME" != "N/A" ] && [ "$OPT3_AVG_TIME" != "N/A" ]; then
    IMPROVEMENT=$(echo "scale=2; ($OPT2_AVG_TIME - $OPT3_AVG_TIME) / $OPT2_AVG_TIME * 100" | bc)
    log_result "🚀 optimization_level=3 improved performance by ${IMPROVEMENT}%"
fi

echo ""
log_info "🎉 Comparison completed!"
