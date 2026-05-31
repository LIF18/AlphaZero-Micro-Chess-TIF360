#! /bin/bash

#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /data/users/zhenqil/.venv/bin/activate

python3 -u tokenize_dataset2.py