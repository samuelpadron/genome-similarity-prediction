#! /usr/bin/env bash

# change working directory to folder where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# set environment variables
source ../.env 2> /dev/null || source .env

# determine where to download to
TARGET_DIR=$SCRIPT_DIR/../logs/cluster
mkdir -p "$TARGET_DIR"

# rsync remote logs
USERNAME=spadronalcala
rsync -azP "$USERNAME"@cn99.science.ru.nl:/ceph/csedu-scratch/users/$USERNAME/"$POJECT_NAME"/logs "$TARGET_DIR"
