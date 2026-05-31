import torch
import json
import random
import math
import chess
import chess.engine
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# --- Configuration Parameters ---
INPUT_DATA_PATH = "tokenized_data/dataset_5M.pt"
OUTPUT_DATA_PATH = "tokenized_data/dataset_distilled.pt"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_SAMPLES = 2000000        # Extract 2 million positions
STOCKFISH_DEPTH = 10         # Depth
TOP_K_MOVES = 3              # Extract Top 3 moves
CHUNK_SIZE = 500             # Treat every batch of games as a chunk, completely eliminating deadlocks!

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def cp_to_win_prob(cp_score):
    if cp_score is None: return 0.0
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp_score)) - 1.0

# Complete logic for processing a chunk (batch)
def process_chunk(chunk_data):
    token2id = load_vocab()
    pad_id = token2id.get("<pad>")
    bos_id = token2id.get("<s>")
    id2token = {v: k for k, v in token2id.items()}
    
    # Explicitly start the engine at the beginning of the batch
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    valid_results = []
    
    try:
        # Process each game within this batch
        for sequence in chunk_data:
            seq_list = [tok for tok in sequence.tolist() if tok not in (pad_id, bos_id, token2id.get("</s>"))]
            if len(seq_list) < 20: continue
                
            trunc_point = random.randint(10, len(seq_list) - 5)
            context_ids = seq_list[:trunc_point]
            
            board = chess.Board()
            try:
                for move_id in context_ids:
                    board.push_san(id2token[move_id])
            except Exception:
                continue 
                
            if board.is_game_over(): continue
                
            try:
                info = engine.analyse(board, chess.engine.Limit(depth=STOCKFISH_DEPTH), multipv=TOP_K_MOVES)
            except Exception:
                continue
                
            if not info: continue
                
            best_info = info[0]
            score_obj = best_info["score"].white() if board.turn == chess.WHITE else best_info["score"].black()
            value_target = cp_to_win_prob(score_obj.score(mate_score=10000))
            
            policy_targets = {}
            cp_scores = []
            moves = []
            
            for pv_info in info:
                if "pv" in pv_info and len(pv_info["pv"]) > 0:
                    move = pv_info["pv"][0]
                    san_str = board.san(move)
                    if san_str in token2id:
                        s_obj = pv_info["score"].white() if board.turn == chess.WHITE else pv_info["score"].black()
                        cp = s_obj.score(mate_score=10000)
                        cp_scores.append(cp / 100.0) 
                        moves.append(token2id[san_str])
                        
            if not moves: continue
                
            max_cp = max(cp_scores)
            exp_scores = [math.exp(cp - max_cp) for cp in cp_scores]
            sum_exp = sum(exp_scores)
            
            for m, e in zip(moves, exp_scores):
                policy_targets[m] = e / sum_exp

            valid_results.append({
                "input_ids": [bos_id] + context_ids,
                "value_target": value_target,
                "policy_targets": policy_targets
            })
            
    finally:
        # Forcefully close the engine to release resources after processing the batch! Kill deadlocks!
        engine.quit()

    # Return the data volume of this batch and successfully extracted valid samples
    return len(chunk_data), valid_results


def main():
    print(f"📦 Loading original 5M dataset (file is approx. 5GB+, please wait patiently)...")
    full_data = torch.load(INPUT_DATA_PATH)
    
    sample_indices = random.sample(range(len(full_data)), NUM_SAMPLES)
    sampled_data = [full_data[i] for i in sample_indices]
    
    # Split 2 million data points into small chunks
    chunks = [sampled_data[i:i + CHUNK_SIZE] for i in range(0, len(sampled_data), CHUNK_SIZE)]
    
    print(f"🚀 Launching parallel data distillation, split into {len(chunks)} batches, all CPU cores firing!")
    distilled_dataset = []
    
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        # Submit all batch chunks
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        
        # Progress bar still maintains real-time monitoring of 2 million games
        with tqdm(total=NUM_SAMPLES, desc="🔥 Deep Special Training Progress") as pbar:
            for future in as_completed(futures):
                try:
                    chunk_len, res = future.result()
                    distilled_dataset.extend(res)
                    pbar.update(chunk_len)  # Progress bar advances by chunk length after each batch finishes
                except Exception as e:
                    print(f"\nA batch failed to process: {e}")
                    
    print(f"\n✅ Distillation completed successfully! No deadlocks! Successfully generated {len(distilled_dataset)} high-quality samples.")
    print(f"💾 Saving to {OUTPUT_DATA_PATH} ...")
    torch.save(distilled_dataset, OUTPUT_DATA_PATH)
    print("🎉 Saved successfully! Go prepare for the Value Head training!")

if __name__ == "__main__":
    main()