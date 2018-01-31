#!/usr/bin/env bash

source ./docker_settings.sh


docker run \
    -it \
    -p 8888:8888 \
    -v ${PWD}:${THIS_FOLDER_IN_CONTAINER} \
    ${DOCKER_IMAGE}
