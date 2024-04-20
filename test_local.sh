#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

# Create an output directory if it doesn't exist
OUTPUT_DIR="$SCRIPTPATH/output"
mkdir -p "$OUTPUT_DIR"

# Run the Docker container, mounting the input directory and the output directory
docker run --cpus=4 --memory=32gb --shm-size=32gb --gpus='device=0' --rm \
        -v "$SCRIPTPATH"/test/input:/input/ \
        -v "$OUTPUT_DIR:/output/" \
        acouslicai_baseline

# No need to remove a Docker volume since we're using a bind mount
