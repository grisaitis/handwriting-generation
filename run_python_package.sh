#!/usr/bin/env bash

source ./docker_settings.sh

PYTHON_PACKAGE=$1

docker run \
    -it \
    -v ${PWD}:${THIS_FOLDER_IN_CONTAINER} \
    -w ${THIS_FOLDER_IN_CONTAINER} \
    ${DOCKER_IMAGE} \
    python -m ${PYTHON_PACKAGE}
