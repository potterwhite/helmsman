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
# Automated Deploy and Test Script
# ==============================================================================
# This script automates the entire workflow:
#   1. Build C++ code for rk3588s
#   2. Rsync binaries to target board
#   3. Execute inference remotely
#   4. Fetch and display results
# ==============================================================================

set -e

# --- Configuration ---
REMOTE_HOST="evboard"
REMOTE_USER="root"
REMOTE_PASSWORD="123"
REMOTE_BASE="/usr"
REMOTE_TEST_DIR="/root"
LOCAL_INSTALL_DIR="runtime/cpp/install/rk3588s/release"
TEST_IMAGE="${REMOTE_TEST_DIR}/green-fall-girl-point-to.png"
TEST_MODEL="${REMOTE_TEST_DIR}/modnet_int8.rknn"
TEST_OUTPUT_DIR="${REMOTE_TEST_DIR}/debug/cpp/"

# Colors
GREEN='\033[1;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[1;31m'
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

# ==============================================================================
# Step 1: Build C++ Code
# ==============================================================================
log_step "Building C++ code for rk3588s..."
./helmsman build cpp build rk3588s

log_step "Installing binaries..."
./helmsman build cpp install rk3588s

# ==============================================================================
# Step 2: Rsync to Target Board
# ==============================================================================
log_step "Syncing binaries to ${REMOTE_HOST}:${REMOTE_BASE}..."
sshpass -p "${REMOTE_PASSWORD}" rsync -avh --progress "${LOCAL_INSTALL_DIR}/" "${REMOTE_HOST}:${REMOTE_BASE}/"

if [ $? -eq 0 ]; then
    log_info "Rsync completed successfully"
else
    log_error "Rsync failed"
    exit 1
fi

# ==============================================================================
# Step 3: Execute Inference Remotely
# ==============================================================================
log_step "Executing inference on ${REMOTE_HOST}..."

# Create output directory if it doesn't exist
sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "mkdir -p ${TEST_OUTPUT_DIR}"

# Run inference and capture output
log_info "Running: Helmsman_Matting_Client ${TEST_IMAGE} ${TEST_MODEL} ${TEST_OUTPUT_DIR}"

sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "cd ${REMOTE_TEST_DIR} && Helmsman_Matting_Client ${TEST_IMAGE} ${TEST_MODEL} ${TEST_OUTPUT_DIR}" 2>&1 | tee /tmp/remote_inference.log

INFERENCE_EXIT_CODE=${PIPESTATUS[0]}

# ==============================================================================
# Step 4: Check Results
# ==============================================================================
if [ ${INFERENCE_EXIT_CODE} -eq 0 ]; then
    log_info "✅ Inference completed successfully!"

    # List output files
    log_step "Output files generated:"
    sshpass -p "${REMOTE_PASSWORD}" ssh "${REMOTE_HOST}" "ls -lh ${TEST_OUTPUT_DIR}"

else
    log_error "❌ Inference failed with exit code ${INFERENCE_EXIT_CODE}"
    log_warn "Check the log above for details"
    exit 1
fi

log_info "🎉 All steps completed successfully!"
