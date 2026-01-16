
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


# # remove docker container
# sudo docker rm -f rlinf