#! /bin/bash
source container_setup/credentials

docker run \
    -d \
    --shm-size=8g \
    --memory=8g \
    --cpuset-cpus=0-7 \
    --user ${DOCKER_USER_ID}:${DOCKER_GROUP_ID} \
    --name ${CONTAINER_NAME} \
    --rm \
    -it \
    --init \
	--gpus '"device=0,1,2"' \
    -v ${SRC}:/app \
    -p ${INNER_PORT}:${CONTAINER_PORT} \
    ${DOCKER_NAME} \
    bash