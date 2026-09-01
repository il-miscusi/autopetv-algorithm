#!/bin/bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Check and download weights if necessary
echo "Checking model weights..."
bash "$SCRIPTPATH/check_weights.sh"
if [ $? -ne 0 ]; then
    echo "ERROR: Weight check failed. Cannot proceed with build."
    exit 1
fi

echo ""
echo "Building Docker image..."
docker build -t autopet_baseline "$SCRIPTPATH"
