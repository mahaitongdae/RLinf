#!/bin/bash
#
# Test script for DDP actor worker with OpenPi model.
# Tests forward and backward pass with mock data.
#
# Usage:
#   ./run_test_ddp_openpi.sh [NUM_GPUS] [FORWARD_TYPE]
#
# Example:
#   ./run_test_ddp_openpi.sh 1                    # Single GPU, default_forward
#   ./run_test_ddp_openpi.sh 1 default_forward    # Single GPU, default_forward
#   ./run_test_ddp_openpi.sh 1 awr_forward        # Single GPU, awr_forward
#   ./run_test_ddp_openpi.sh 2 awr_forward        # 2 GPUs, awr_forward
#

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

# Number of GPUs (default: 1)
NUM_GPUS=${1:-1}

# Forward type (default: default_forward, options: default_forward, awr_forward)
FORWARD_TYPE=${2:-default_forward}

# Environment setup
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Vulkan/EGL setup (for Lambda Cloud compatibility)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "============================================================"
echo "DDP OpenPi Forward/Backward Test"
echo "============================================================"
echo "REPO_PATH: ${REPO_PATH}"
echo "NUM_GPUS: ${NUM_GPUS}"
echo "FORWARD_TYPE: ${FORWARD_TYPE}"
echo "Python: $(which python)"
echo "============================================================"

# Run test
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU - can run without torchrun
    python ${SCRIPT_DIR}/test_ddp_openpi_forward_backward.py \
        --config-path ${SCRIPT_DIR} \
        --config-name test_ddp_openpi \
        algorithm.forward_type=${FORWARD_TYPE}
else
    # Multi-GPU - use torchrun
    torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=29500 \
        ${SCRIPT_DIR}/test_ddp_openpi_forward_backward.py \
        --config-path ${SCRIPT_DIR} \
        --config-name test_ddp_openpi \
        actor.global_batch_size=$((8 * NUM_GPUS)) \
        algorithm.forward_type=${FORWARD_TYPE}
fi

echo ""
echo "Test completed!"

