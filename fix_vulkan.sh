#!/bin/bash

# Define paths and content
ICD_PATH="/usr/share/vulkan/icd.d/nvidia_icd.json"
EGL_PATH="/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
LAYER_PATH="/etc/vulkan/implicit_layer.d/nvidia_layers.json"

ICD_CONTENT='{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155"
    }
}'

EGL_CONTENT='{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}'

LAYER_CONTENT='{
    "file_format_version" : "1.0.0",
    "layer": {
        "name": "VK_LAYER_NV_optimus",
        "type": "INSTANCE",
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.2.155",
        "implementation_version" : "1",
        "description" : "NVIDIA Optimus layer",
        "functions": {
            "vkGetInstanceProcAddr": "vk_optimusGetInstanceProcAddr",
            "vkGetDeviceProcAddr": "vk_optimusGetDeviceProcAddr"
        },
        "enable_environment": {
            "__NV_PRIME_RENDER_OFFLOAD": "1"
        },
        "disable_environment": {
            "DISABLE_LAYER_NV_OPTIMUS_1": ""
        }
    }
}'

# Check for root privileges
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root (use sudo)"
   exit 1
fi

echo "--- Starting Vulkan Environment Check ---"

# 1. Run vulkaninfo to check status
# if vulkaninfo >/dev/null 2>&1; then
#     echo "[OK] vulkaninfo reports a valid configuration."
# else
#     echo "[!] vulkaninfo failed. Attempting to fix configuration files..."

# 2. Check and fix nvidia_icd.json
if [ ! -f "$ICD_PATH" ]; then
    echo "[*] Creating $ICD_PATH..."
    mkdir -p "$(dirname "$ICD_PATH")"
    echo "$ICD_CONTENT" > "$ICD_PATH"
else
    echo "[OK] $ICD_PATH exists."
fi

# 3. Check and fix 10_nvidia.json
if [ ! -f "$EGL_PATH" ]; then
    echo "[*] $EGL_PATH missing. Attempting to install libglvnd-dev..."
    apt-get update && apt-get install -y libglvnd-dev
    
    # Verify if install created it; if not, create manually
    if [ ! -f "$EGL_PATH" ]; then
        echo "[*] Manually creating $EGL_PATH..."
        mkdir -p "$(dirname "$EGL_PATH")"
        echo "$EGL_CONTENT" > "$EGL_PATH"
    fi
else
    echo "[OK] $EGL_PATH exists."
fi

# 4. Check and fix nvidia_layers.json (Optional/A100)
if [ ! -f "$LAYER_PATH" ]; then
    echo "[*] Creating $LAYER_PATH (recommended for A100/Optimus)..."
    mkdir -p "$(dirname "$LAYER_PATH")"
    echo "$LAYER_CONTENT" > "$LAYER_PATH"
else
    echo "[OK] $LAYER_PATH exists."
fi

echo "--- Fixes applied. Re-testing vulkaninfo ---"
if vulkaninfo >/dev/null 2>&1; then
    echo "[SUCCESS] Vulkan is now working."
else
    echo "[ERROR] vulkaninfo still failing. Check NVIDIA driver installation."
fi
# fi