#!/bin/bash -e
#SBATCH --account=cseduproject
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --qos=csedu-small
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --mail-user=samuel.padronalcala@ru.nl
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --mail-type=END

project_dir=.

source "$project_dir"/venv/bin/activate
python -m train wandb=null experiment=hg38/pair_alignment_load_finetune_model model.fused_dropout_add_ln=False train.pretrained_model_path=/scratch/spadronalcala/hyenadna-small-32k-seqlen/weights.ckpt