nano test_job.sh
sbatch test_job.sh
ls
cat slurm-180323.out
pwd
tensorboard --logdir ./chess_model_checkpoints_5M/runs --bind_all
tensorboard --logdir ./runs/alphazero_finetune --bind_all
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install python-chess
python -V
pytorch -V
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install beautifulsoup4
sbatch run_download.sh
squeue -j 204470
cat slurm-204470.out
scancel 204470
sbatch run_download.sh
squeue -j 204471
cat slurm-204471.out
sbatch run_tokenize.sh
squeue -j 204476
cat slurm-204476.out
squeue -j 204476
cat slurm-204476.out
squeue -j 204476
scancel 204476
pwd
sbatch run_tokenize.sh
squeue -j 204479
squeue -j 204497
scancel 204497
sbatch run_tokenize.sh
squeue -j 204499
scancel 204499
sbatch run_tokenize.sh
squeue -j 204501
scancel 204501
sbatch run_tokenize.sh
squeue -j 204502
cat slurm-204502.out
squeue -j 204502
sstat -j 204502 --format=JobID,AveCPU,MaxRSS
squeue -j 204502
sstat -j 204502 --format=JobID,AveCPU,MaxRSS
sinfo -n callisto -O NodeList,CPULoad,Memory
sacct -j 204502 --format=JobID,JobName,State,Elapsed,AllocCPUS
squeue -j 204502
scancel 204502
sbatch run_tokenize.sh
squeue -j 204505
scancel 204502
scancel 204505
sbatch run_tokenize.sh
squeue -j 204512
scancel 204512
sbatch run_tokenize.sh
squeue -j 204515
sbatch run_slice.sh
squeue -j 204598
pip install transformers
sbatch run_train.sh
squeue -j 204602
squeue -p long
squeue -j 204602 --start
squeue -p long
squeue -w uranus
squeue -w neptune
squeue -p long
sbatch run_train.sh
squeue -j 204604
squeue -p long
scancel 204602
squeue -p long
sbatch run_train.sh
squeue -j 204633
squeue -p long
sbatch run_slice.sh
squeue -j 204636
squeue -p long
sbatch run_train.sh
scancel 204644
scancel 204633
squeue -p long
squeue -j 204638
squeue -p long
pip install "accelerate>=1.1.0" tensorboard
sbatch run_train.sh
squeue -p long
sbatch run_train.sh
squeue -p long
sbatch run_train.sh
squeue -j 204649
sbatch run_train.sh
squeue -p long
mkdir stockfish_engine && cd stockfish_engine
wget https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-ubuntu-x86-64-avx2.tar
tar -xvf stockfish-ubuntu-x86-64-avx2.tar
cd ..
sbatch run_eval.sh
squeue -j 204745
sbatch mcts_eval.py
sbatch run_mcts_eval.sh
squeue -j 204760
sbatch run_mcts_eval.sh
sbatch run_beam_eval.sh
sbatch run_play.sh
squeue -j 204814
sbatch run_play.sh
sbatch run_dataset_update.sh
squeue -p long
squeue -j 204852
scancel 204852
sbatch run_dataset_update.sh
squeue -j 204877
scancel 204877
sbatch run_dataset_update.sh
squeue -p long
sbatch run_train_finetune.sh
squeue -p long

sbatch run_train_finetune.sh
sbatch run_play.sh
sbatch run_dataset_update.sh
squeue -j 205050
scancel 205050
sbatch run_dataset_update.sh
sbatch run_play.sh
sbatch run_train_finetune.sh
squeue -p long
scancel 207408
sbatch run_train_finetune.sh
squeue -p long
squeue -u zhenqil
scancel 207597
sinfo -p long
sbatch run_train_finetune.sh
squeue -p long
scancel 207775
sinfo -p long
sbatch run_train_finetune.sh
squeue -p long
sinfo -p long
scancel 208049
sinfo -p long
sbatch run_train_finetune.sh
sinfo -p long
squeue -p long
sbatch run_play.sh
