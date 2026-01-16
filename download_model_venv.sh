
# hf download gen-robot/openvla-7b-rlvla-warmup \
# --local-dir /workspace/models/openvla-7b-rlvla-warmup

# hf download RLinf/RLinf-Pi0-ManiSkill-25Main-SFT \
# --local-dir /workspace/models/RLinf-Pi0-ManiSkill-25Main-SFT

current_dir=$(pwd)

mkdir -p /.cache/models

hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
--local-dir ~/cache/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

# # Download assets
assets_dir="${current_dir}/rlinf/envs/maniskill/assets"
mkdir -p "${assets_dir}"
if [ ! -w "${assets_dir}" ]; then
    if command -v sudo >/dev/null 2>&1; then
        sudo chown -R "${USER}:${USER}" "${assets_dir}"
    fi
fi
if [ ! -w "${assets_dir}" ]; then
    echo "Error: ${assets_dir} is not writable. Fix permissions and retry."
    exit 1
fi
hf download --repo-type dataset RLinf/maniskill_assets \
--local-dir ${assets_dir}

# bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi

# bash examples/embodiment/run_embodiment.sh libero_spatial_ppo_openpi_quickstart
