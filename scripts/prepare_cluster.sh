#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# load the environment variables from the `.env` file
source ../.env 2> /dev/null || source .env

# make a directory on the ceph file system to store logs and checkpoints
# and make a symlink to access it directly from the root of the project
if [ -z "$CEPH_USER_DIR" ]; then
    echo "Please set $CEPH_USER_DIR in .env file."
    exit
fi
if [ -z "$PROJECT_NAME" ]; then
    echo "Please set $PROJECT_NAME in .env file."
    exit
fi

echo "creating symlinks pointing to $CEPH_USER_DIR to $CEPH_USER_DIR/$PROJECT_NAME"

mkdir -p "$CEPH_USER_DIR"
chmod 700 "$CEPH_USER_DIR" # only you can access
mkdir -p "$CEPH_USER_DIR"/"$PROJECT_NAME"/data "$CEPH_USER_DIR"/"$PROJECT_NAME"/logs

ln -sfn "$CEPH_USER_DIR"/"$PROJECT_NAME"/data "$SCRIPT_DIR"/../data
ln -sfn "$CEPH_USER_DIR"/"$PROJECT_NAME"/logs "$SCRIPT_DIR"/../logs
