#! /bin/bash

#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:L4:8        # Core command: Request 8 L4 graphics cards from the cluster.
#SBATCH --mem=128G
#SBATCH --time=24:00:00

source /data/users/zhenqil/.venv/bin/activate

torchrun --nproc_per_node=8 train_hf.py