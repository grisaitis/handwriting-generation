#!/usr/bin/env bash

source ./docker_settings.sh

docker run \
    -it \
    --rm \
    -v ${PWD}:${THIS_FOLDER_IN_CONTAINER} \
    -v ~/.bash_history:/root/.bash_history \
    -w ${THIS_FOLDER_IN_CONTAINER} \
    ${DOCKER_IMAGE} \
    bash
