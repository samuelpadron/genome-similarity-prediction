#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.err

project_dir=.

source "$project_dir"/venv/bin/activate
echo "$(ls /scratch/spadronalcala)"
python -m my_train