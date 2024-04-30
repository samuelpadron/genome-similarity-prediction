#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:5
#SBATCH --qos=csedu-large
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs
#SBATCH --mail-type=END

project_dir=.
job_id=$SLURM_JOB_ID

# optimization hyperparameters
batch_size=24
learning_rate=6e-4
weight_decay=0.001

source "$project_dir"/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m train_lightning "$job_id" $batch_size $learning_rate $weight_decay