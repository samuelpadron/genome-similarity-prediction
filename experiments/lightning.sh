#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:5
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs

project_dir=.
job_id=$SLURM_JOB_ID

# optimization hyperparameters
batch_size=8
learning_rate=6e-4
weight_decay=0.001

source "$project_dir"/venv/bin/activate
python -m train_lightning "$job_id" $batch_size $learning_rate $weight_decay /vol/csedu-nobackup/project/spadronalcala/pair_alignment/danRer10