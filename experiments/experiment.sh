#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:2
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs
#SBATCH --mail-type=END

project_dir=.
job_id=$SLURM_JOB_ID

# optimization hyperparameters
learning_rate=6e-4
weight_decay=0.1

source "$project_dir"/venv/bin/activate
python -m my_train "$job_id" $learning_rate $weight_decay