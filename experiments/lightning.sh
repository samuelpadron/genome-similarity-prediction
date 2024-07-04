#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs

project_dir=.

source "$project_dir"/venv/bin/activate
python -m train --model.learning_rate=6e-4 --model.weight_decay=0.001 --model.device=cuda  --model.pretrained_model_name=hyenadna-small-32k-seqlen --data.batch_size=64
