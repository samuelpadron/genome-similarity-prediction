#! /usr/bin/env bash

# change working directory to folder where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set environment variables
source ../.env 2> /dev/null || source .env

# determine where to download to
TARGET_DIR=$SCRIPT_DIR/../logs/cluster
mkdir -p "$TARGET_DIR"

# rsync remote logs
echo "line 13 in this script requires your science.ru.nl username"; exit 0 # delete line once you've set the username
echo "line 14 in this script requires your path to the log folder on the cluster"; exit 0 # delete line once you've set the correct path
USERNAME=nvaessen
rsync -azP "$USERNAME"@cn99.science.ru.nl:/ceph/das-scratch/users/$USERNAME/"$PROJECT_NAME"/logs "$TARGET_DIR"