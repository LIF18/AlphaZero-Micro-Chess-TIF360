#! /bin/bash

#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00

source /data/users/zhenqil/.venv/bin/activate

python3 -u slice_data.py