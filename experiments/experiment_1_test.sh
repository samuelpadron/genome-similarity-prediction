#!/bin/bash -e
#SBATCH --partion=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --account=cseduproject
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --mail-type=END,FAIL

echo "This experiment should work"
