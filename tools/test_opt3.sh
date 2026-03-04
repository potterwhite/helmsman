#!/bin/bash
# Test INT8 optimization_level=3 model

set -e

REMOTE_HOST="evboard"
REMOTE_PASSWORD="123"
REMOTE_TEST_DIR="/root"
LOCAL_MODELS_DIR="modnet-models"
TEST_IMAGE="${REMOTE_TEST_DIR}/green-fall-girl-point-to.png"
TEST_OUTPUT_DIR="${REMOTE_TEST_DIR}/debug/cpp/"
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

log_step "Syncing INT8 opt3 model..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh "${LOCAL_MODELS_DIR}/modnet_int8_opt3.rknn" "${REMOTE_HOST}:${INT8_OPT3_MODEL}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

log_benchmark "=========================================="
log_benchmark "Testing INT8 optimization_level=3"
log_benchmark "=========================================="

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Client ${TEST_IMAGE} ${INT8_OPT3_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_opt3_test.log

OPT3_EXIT_CODE=${PIPESTATUS[0]}

if [ ${OPT3_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 opt3 completed"
    OPT3_AVG_TIME=$(grep "Performance Benchmark" /tmp/int8_opt3_test.log | grep -oP '\d+\.\d+(?= ms)' | awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    log_result "INT8 opt3 Average Time: ${OPT3_AVG_TIME} ms"
else
    log_error "❌ INT8 opt3 failed"
    exit 1
fi

echo ""
log_benchmark "=========================================="
log_benchmark "Performance Comparison"
log_benchmark "=========================================="

echo -e "${CYAN}Version${NC}        | ${CYAN}Avg Time (ms)${NC} | ${CYAN}FPS${NC}    | ${CYAN}vs FP16${NC}"
echo "-------------- | ------------- | ------ | --------"

FP16_BASELINE=287
INT8_OPT2_BASELINE=383.37

FP16_FPS=$(echo "scale=2; 1000 / $FP16_BASELINE" | bc)
OPT2_FPS=$(echo "scale=2; 1000 / $INT8_OPT2_BASELINE" | bc)

printf "%-14s | %-13s | %-6s | %-8s\n" "FP16 Baseline" "$FP16_BASELINE" "$FP16_FPS" "1.00x"
printf "%-14s | %-13s | %-6s | %-8s\n" "INT8 opt2" "$INT8_OPT2_BASELINE" "$OPT2_FPS" "0.75x"

if [ "$OPT3_AVG_TIME" != "N/A" ]; then
    OPT3_FPS=$(echo "scale=2; 1000 / $OPT3_AVG_TIME" | bc)
    OPT3_SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $OPT3_AVG_TIME" | bc)
    printf "%-14s | %-13s | %-6s | ${GREEN}%-8s${NC}\n" "INT8 opt3" "$OPT3_AVG_TIME" "$OPT3_FPS" "${OPT3_SPEEDUP}x"

    echo ""
    IMPROVEMENT=$(echo "scale=2; ($INT8_OPT2_BASELINE - $OPT3_AVG_TIME) / $INT8_OPT2_BASELINE * 100" | bc)
    log_result "🚀 optimization_level=3 improved performance by ${IMPROVEMENT}% vs opt2"
fi

echo ""
log_info "🎉 Test completed!"
