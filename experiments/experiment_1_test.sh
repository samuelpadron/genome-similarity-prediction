#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu-prio,csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=END

project_dir=.

source "$project_dir"/venv/bin/activate

cd "$project_dir"/flash-attention/

pip install -e .

cd ..

python -m train wandb=null experiment=hg38/genomic_benchmark_scratch
