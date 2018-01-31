#!/usr/bin/env bash

DOCKER_IMAGE="tensorflow/tensorflow:1.5.0-py3"
THIS_FOLDER=$(basename ${PWD})
THIS_FOLDER_IN_CONTAINER=/notebooks/${THIS_FOLDER}
