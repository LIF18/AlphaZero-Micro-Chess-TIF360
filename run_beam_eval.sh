#! /bin/bash
#SBATCH --partition=short
#SBATCH --gres=gpu:L4:1
#SBATCH --mem=16G

source /data/users/zhenqil/.venv/bin/activate
python3 beam_search_eval.py