#!/bin/bash

# Exit on error
set -e

echo "--- Starting NVIDIA Vulkan/EGL Setup for Lambda Cloud ---"

# 1. Detect NVIDIA Driver Version
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d'.' -f1)

if [ -z "$DRIVER_VERSION" ]; then
    echo "Error: NVIDIA driver not detected. Please ensure nvidia-smi works."
    exit 1
fi

echo "Detected NVIDIA Driver Version: $DRIVER_VERSION"

# 2. Install necessary libraries
echo "Installing Vulkan and EGL libraries..."
sudo apt-get update
sudo apt-get install -y \
    libvulkan1 \
    libnvidia-gl-570-server \
    libglvnd-dev \
    vulkan-tools

# 3. Create a persistent configuration for the shell (Zsh)
ZSHRC="$HOME/.bashrc"
echo "Updating $ZSHRC with environment variables..."

# Function to add export if not already present
add_to_zshrc() {
    grep -qF "$1" "$ZSHRC" || echo "$1" >> "$ZSHRC"
}

add_to_zshrc 'export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json'
add_to_zshrc 'export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
add_to_zshrc 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH'

# 4. Export variables to the current session immediately
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

echo "--- Setup Complete ---"

# 5. Verification
echo "Verifying Vulkan device..."
vulkaninfo | grep deviceName || echo "Warning: vulkaninfo failed. Check your driver installation."

echo "Please run 'source ~/.zshrc' to update your current terminal."