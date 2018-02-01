#!/usr/bin/env bash

source ./docker_settings.sh


docker run \
    -it \
    -p 8888:8888 \
    -p 6006:6006 \
    -v ${PWD}:${THIS_FOLDER_IN_CONTAINER} \
    -w /notebooks \
    ${DOCKER_IMAGE} \
    /run_jupyter.sh --allow-root
