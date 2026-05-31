## 📂 Repository Structure & Pipeline

This repository contains the complete end-to-end pipeline for the project, ranging from data acquisition to the final AlphaZero-style Beam Search evaluation.

The codebase is highly optimized for High-Performance Computing (HPC) clusters using the **SLURM workload manager**. Each core Python script is paired with a corresponding `.sh` batch script for streamlined job submission.

### Phase 0: Data Acquisition & Preprocessing

Scripts responsible for fetching, cleaning, and tokenizing raw human chess data into ML-ready PyTorch tensors.

- **`download_elite.py`** * 🚀 **Execution:** `sbatch run_download.sh`
  - **Description:** Automated web scraper that securely downloads high-quality Grandmaster PGN archives from the Lichess Elite database, featuring anti-scraping headers and resume-download capabilities.
- **`tokenize_dataset2.py`** * 🚀 **Execution:** `sbatch run_tokenize.sh` (Allocates 16 CPU cores)
  - **Description:** Industrial-grade data parser. It utilizes a multiprocessing pool to read PGN files, rigorously verifies move legality using the `python-chess` engine, filters games below 2400 ELO, and generates the `vocab.json` dictionary ($\approx$ 14,459 tokens) before compiling fixed-length PyTorch tensors.
- **`slice_data.py`** * 🚀 **Execution:** `sbatch run_slice.sh`
  - **Description:** Memory management script. Safely loads the massive 21GB full dataset tensor and slices it into the core 5M-game training set, a 100k debug set, and a 10k independent evaluation set.

### Phase 1: Pure Policy Pre-training (System 1 Intuition)

- **`train_hf.py`** * 🚀 **Execution:** `sbatch run_train.sh` (Requests 8x NVIDIA L4 GPUs)
  - **Description:** The System 1 pre-training script. Uses Hugging Face `Trainer` and PyTorch Distributed Data Parallel (DDP) to train a micro-scale GPT-2 model ($ \approx $ 15M parameters) on 5 million games via standard autoregressive Next-Token Prediction.

### Phase 2 & 3: Architecture Surgery & Offline Distillation

- **`prepare_distillation.py`** * 🚀 **Execution:** `sbatch run_dataset_update.sh`
  - **Description:** Generates the offline distillation dataset. Prompts the Stockfish engine (Depth 10) to evaluate 200,000 mid-game states, extracting objective win-rates and soft-policy distributions. Utilizes a chunk-based processing pipeline to completely prevent C++ engine thread deadlocks.
- **`train_finetune.py`** * 🚀 **Execution:** `sbatch run_train_finetune.sh` (Requests 4x NVIDIA L4 GPUs)
  - **Description:** Performs the architecture surgery. Freezes the pre-trained Transformer backbone, attaches a newly initialized MLP Value Head, and fine-tunes the network using a multi-task objective ($L = L_{MSE} + L_{CE}$) on the distilled dataset.

### Phase 4: System 2 Inference & Empirical Evaluation

- **`alphazero_beam_play.py`** * 🚀 **Execution:** `sbatch run_play.sh`
  - **Description:** **The ultimate inference engine of this project.** Merges a Depth-3 Beam Search with the AlphaZero PUCT formula ($C=0.5$). The model dynamically explores future trajectories, evaluates leaf nodes using the Value Head, applies strict perspective inversion, and plays against a 1350 ELO Stockfish to demonstrate its defensive capabilities.

### Evaluation & Baselines

Scripts used to establish the metrics and baselines discussed in the *Results and Discussion* section of our report.

- **`inference_eval.py`** (Execution: `sbatch run_eval.sh`)
  - Tests the pure System 1 intuitive accuracy. Evaluates the LLM's next-token prediction against Stockfish Depth-15 "absolute truth" on 1,000 endgame positions, establishing the 31.5% Top-1 accuracy baseline.
- **`play_stockfish.py`** * Baseline empirical matchup. Pits the pure GPT-2 sequence model (System 1 without explicit search) against Stockfish, demonstrating the catastrophic 0% win rate caused by out-of-distribution (OOD) compounding errors.

### Evolutionary & Experimental Scripts

These scripts document our research journey, including ablation studies and alternative methodologies that ultimately justified our final architecture.

- **`mcts_eval.py`** (Execution: `sbatch run_mcts_eval.sh`)
  - **Monte Carlo Tree Search (MCTS) Experiment.** Attempts to use standard MCTS guided by the LLM's prior probabilities. Confirmed that Python-loop overheads make full MCTS unviable for our compute budget, justifying the shift to a lightweight PUCT Beam Search.
- **`beam_search_eval.py`** (Execution: `sbatch run_beam_eval.sh`)
  - Tests multi-ply Beam Search *without* a Value Head (pure sequence probability accumulation) to isolate the impact of the Value Head.
- **`alphazero_play.py`**
  - An early AlphaZero implementation utilizing the Value Head but restricted to Depth-1 greedy search. Highlighted the horizon effect and the necessity for multi-ply look-ahead.

### Phase 5: Visualization & Reporting

This module contains utilities for extracting data from raw HPC cluster logs and generating high-quality, publication-ready academic figures used directly in the final project report and poster presentation.

- **`plot_loss.py`**
  - **Description:** Academic figure generation script. Utilizing `matplotlib` and `seaborn`, it parses the Multi-task Loss (MSE + Cross-Entropy) data extracted from the SLURM training logs to generate a 300 DPI ultra-high-resolution training convergence curve (`loss_curve_for_poster.png`). This chart provides empirical evidence of stable model convergence during the offline knowledge distillation phase, confirming the absence of gradient explosions.
- **`slurm-204956.out`** (and other `.out` SLURM logs)
  - **Description:** Raw standard output logs generated by the Chalmers Minerva cluster. These files meticulously record the detailed loss metrics every 50 steps, learning rate decay, and system-level resource scheduling during the architecture surgery and fine-tuning phase (`train_finetune.py`). They serve as the absolute ground-truth data source for all visualization plots.