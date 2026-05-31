#! /bin/bash

#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --time=04:00:00

source ./ .venv/bin/activate
python3 download_elite.py