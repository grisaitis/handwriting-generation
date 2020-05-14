#!/usr/bin/env bash

source ./docker_settings.sh

CONTAINER_NAME=train-$(basename $(pwd))

echo $CONTAINER_NAME

docker run \
    -d \
    --name $CONTAINER_NAME \
    -v ${PWD}:${THIS_FOLDER_IN_CONTAINER} \
    -w ${THIS_FOLDER_IN_CONTAINER} \
    ${DOCKER_IMAGE} \
    python train.py --rnn_size 50
