
# pull docker image
sudo docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

# start docker container
sudo docker run -it --gpus all \
   --shm-size 100g \
   --net=host \
   --name rlinf \
   -e NVIDIA_DRIVER_CAPABILITIES=all \
   -v ./:/workspace/RLinf \
   -v ~/.cache/models:/workspace/models \
   rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

# remove docker container
sudo docker rm -f rlinf

hf download gen-robot/openvla-7b-rlvla-warmup \
--local-dir /workspace/models/openvla-7b-rlvla-warmup

hf download RLinf/RLinf-Pi0-ManiSkill-25Main-SFT \
--local-dir /workspace/models/RLinf-Pi0-ManiSkill-25Main-SFT

hf download RLinf/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT \
--local-dir /workspace/models/RLinf-Pi0-LIBERO-Spatial-Object-Goal-SFT

# # Download assets
# hf download --repo-type dataset RLinf/maniskill_assets \
# --local-dir /workspace/RLinf/rlinf/envs/maniskill/assets

bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi
