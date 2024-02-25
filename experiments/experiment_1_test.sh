#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu-prio,csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=experiment_1_test_%j.out
#SBATCH --error=experiment_1_test_%j.err
#SBATCH --mail-type=END,FAIL

nvidia-smi
