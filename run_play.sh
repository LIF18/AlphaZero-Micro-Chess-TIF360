#! /bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:L4:1
#SBATCH --mem=16G

source /data/users/zhenqil/.venv/bin/activate
python3 alphazero_beam_play.py