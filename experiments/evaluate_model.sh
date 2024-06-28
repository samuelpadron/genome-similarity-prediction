#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=4:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.errs

project_dir=.
job_id=$SLURM_JOB_ID

source "$project_dir"/venv/bin/activate
python -m test_model "$job_id" lightning_logs/version_4854615/version_0/checkpoints/epoch=99-step=56000.ckpt /vol/csedu-nobackup/project/spadronalcala/pair_alignment/galGal6