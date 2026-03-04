#!/bin/bash
# Copyright (c) 2026 PotterWhite
# Quick deployment script for INT8 model testing (no rebuild)

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
INT8_MODEL="${REMOTE_TEST_DIR}/modnet_int8.rknn"

# Colors
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
CYAN='\033[1;36m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[Deploy]${NC} $1"; }
log_step() { echo -e "${BLUE}[Step]${NC} $1"; }
log_error() { echo -e "${RED}[Error]${NC} $1"; }
log_benchmark() { echo -e "${CYAN}[Benchmark]${NC} $1"; }

# ==============================================================================
# Step 1: Rsync INT8 Model to Target Board
# ==============================================================================
log_step "Syncing INT8 model to ${REMOTE_HOST}..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh --progress "${LOCAL_MODELS_DIR}/modnet_int8.rknn" "${REMOTE_HOST}:${INT8_MODEL}"

if [ $? -eq 0 ]; then
    log_info "INT8 model synced successfully (25MB)"
else
    log_error "INT8 model sync failed"
    exit 1
fi

# ==============================================================================
# Step 2: Create Output Directory
# ==============================================================================
sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

# ==============================================================================
# Step 3: Run INT8 Inference Benchmark
# ==============================================================================
log_benchmark "=========================================="
log_benchmark "Testing INT8 Model Performance"
log_benchmark "=========================================="

log_info "Running: Helmsman_Matting_Client ${TEST_IMAGE} ${INT8_MODEL} ${TEST_OUTPUT_DIR}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Client ${TEST_IMAGE} ${INT8_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/int8_inference.log

INT8_EXIT_CODE=${PIPESTATUS[0]}

if [ ${INT8_EXIT_CODE} -eq 0 ]; then
    log_info "✅ INT8 inference completed successfully!"

    # Extract performance metrics
    echo ""
    log_benchmark "=========================================="
    log_benchmark "Performance Analysis"
    log_benchmark "=========================================="

    echo -e "${CYAN}Extracting inference times from log...${NC}"
    grep "Performance Benchmark" /tmp/int8_inference.log | awk '{print $NF}' | sed 's/ms.//' > /tmp/int8_times.txt

    INT8_AVG_TIME=$(awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}' /tmp/int8_times.txt)
    INT8_MIN_TIME=$(sort -n /tmp/int8_times.txt | head -1)
    INT8_MAX_TIME=$(sort -n /tmp/int8_times.txt | tail -1)

    if [ "$INT8_AVG_TIME" != "N/A" ]; then
        INT8_FPS=$(echo "scale=2; 1000 / $INT8_AVG_TIME" | bc)

        echo ""
        echo -e "${CYAN}INT8 Quantized Model Results:${NC}"
        echo "  Average Time: ${GREEN}${INT8_AVG_TIME} ms${NC}"
        echo "  Min Time:     ${INT8_MIN_TIME} ms"
        echo "  Max Time:     ${INT8_MAX_TIME} ms"
        echo "  FPS:          ${GREEN}${INT8_FPS}${NC}"
        echo ""

        # Compare with baseline (FP16 was ~300ms)
        FP16_BASELINE=300
        SPEEDUP=$(echo "scale=2; $FP16_BASELINE / $INT8_AVG_TIME" | bc)
        FP16_FPS=$(echo "scale=2; 1000 / $FP16_BASELINE" | bc)

        echo -e "${CYAN}Comparison with FP16 Baseline (~300ms):${NC}"
        echo "  FP16 FPS:     ${FP16_FPS}"
        echo "  INT8 FPS:     ${GREEN}${INT8_FPS}${NC}"
        echo "  Speedup:      ${GREEN}${SPEEDUP}x${NC}"
        echo "  Improvement:  ${GREEN}$(echo "scale=1; ($INT8_FPS - $FP16_FPS) / $FP16_FPS * 100" | bc)%${NC}"
    fi
else
    log_error "❌ INT8 inference failed with exit code ${INT8_EXIT_CODE}"
    exit 1
fi

echo ""
log_step "Output files generated:"
sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "ls -lh ${TEST_OUTPUT_DIR}"

echo ""
log_info "🎉 INT8 benchmark completed!"
log_info "📝 Full log saved to: /tmp/int8_inference.log"
