#!/bin/bash

echo "Launching nvidia's pytorch container"
IMAGE="nvcr.io/nvidia/pytorch:20.03-py3"
if [ -z "$1" ]
then
CONTAINER="harmony"
else
CONTAINER="$1"
fi

echo "Run container(${CONTAINER}) from image(${IMAGE})"
nvidia-docker run \
 -it --rm --name $CONTAINER --ipc=host --net=host --privileged \
 --memory=375g --memory-swap=375g --memory-swappiness=0 --memory-reservation=375g --shm-size=375g \
 --ulimit memlock=375000000000:375000000000 \
 --gpus '"device=0,1,2,3"' \
 -v /data:/data -v /workspace:/workspace \
 $IMAGE /bin/bash

# NOTE:
# --privileged: With this mode, we can bind mem and CPU nodes inside the container and use numa effectively.
# --cpuset-cpus: limits which cores (or logical processor in case of hyperthread) your container can run on. --cpuset-cpus=0-1 means that your container can only use logical processor 0 and 1, and no other. When using it, make sure to pair up hyperthreads together.
# --cpuset-mems: NUMA memory node 
# --ulimit: <item type>=<soft limit>[:<hard limit>]. If no ulimits are set, they are inherited from the default ulimits set on the daemon. Docker doesn’t perform any byte conversion. E.g. (https://stackoverflow.com/questions/36298913/docker-container-does-not-inherit-ulimit-from-host)