#! /bin/bash

#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:L4:4 
#SBATCH --mem=64G               
#SBATCH --time=8:00:00          

source /data/users/zhenqil/.venv/bin/activate

torchrun --nproc_per_node=4 train_finetune.py