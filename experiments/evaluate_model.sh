#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=7:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs

project_dir=.

source "$project_dir"/venv/bin/activate
python -m test_model lightning_logs/version_4464002/checkpoints/epoch=99-step=1200.ckpt /vol/csedu-nobackup/project/spadronalcala/pair_alignment/panTro6