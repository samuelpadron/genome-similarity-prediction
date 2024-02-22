#! /usr/bin/env bash
set -e

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# set environment variables
source ../.env 2> /dev/null || source .env

# default directory to save files in
DIR="$SCRIPT_DIR"/../data/pair_aligments
mkdir -p "$DIR"

# download tar file
curl -LJ0 https://raw.githubusercontent.com/samuelpadron/DNA_alignment_similarity_CSV/pair_alignment/true.csv -o "$DIR/true.csv"
curl -LJ0 https://raw.githubusercontent.com/samuelpadron/DNA_alignment_similarity_CSV/pair_alignment/false.csv -o "$DIR/false.csv"
# extract tar file
tar xfv "$DIR"/cifar-10-python.tar.gz -C "$DIR"
