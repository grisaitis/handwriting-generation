#!/usr/bin/env bash

DOCKER_IMAGE="tensorflow/tensorflow:1.5.0-py3"
THIS_FOLDER=$(basename ${PWD})

docker run \
    -it \
    -p 8888:8888 \
    -v ${PWD}:/notebooks/${THIS_FOLDER} \
    ${DOCKER_IMAGE}

#    -v ${HOME}/src/aidemoi:/opt/aidemoi \
