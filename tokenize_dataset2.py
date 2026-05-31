import os
import zipfile
import io
import json
from collections import Counter
from multiprocessing import Pool, cpu_count
import chess.pgn
import torch
import gc

# Configuration path and parameters
DATA_DIR = "lichess_elite_dataset"
OUTPUT_DIR = "tokenized_data"
VOCAB_FILE = "vocab.json"
MAX_MOVES = 128  # Maximum number of steps to cut/fill (counted as half a step, 128 steps or 64 rounds)
MIN_ELO = 2400  # Minimum average Elo rating for filtering games

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Predefined special tokens
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]


def process_zip_archive(zip_filename):
    zip_path = os.path.join(DATA_DIR, zip_filename)
    games_san = []
    local_vocab = set()
    valid_game_count = 0
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.pgn'):
                    print(f"[{zip_filename}] assigned to the core, begin secure stream parsing....")
                    with z.open(file_info) as f:
                        stream = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
                        
                        while True:
                            try:
                                game = chess.pgn.read_game(stream)
                            except Exception:
                                continue
                                
                            if game is None:
                                break
                            
                            # Print the survival status every 50,000 games.
                            valid_game_count += 1
                            if valid_game_count % 50000 == 0:
                                print(f"[{zip_filename}] In-depth analysis continues... {valid_game_count} games have been scanned")

                            # Check the Elo threshold
                            try:
                                white_elo = int(game.headers.get("WhiteElo", 0))
                                black_elo = int(game.headers.get("BlackElo", 0))
                                if (white_elo + black_elo) / 2 < MIN_ELO:
                                    continue
                            except (ValueError, TypeError):
                                continue
                                
                            board = game.board()
                            san_moves = []
                            is_game_corrupted = False
                            
                            # Crucial Defense: Move-by-Move Verification
                            # The original `game.mainline_moves()` is a generator; it will crash if it encounters a bad move in the middle.
                            # We must manually traverse its nodes to catch the `ValueError` (illegal sanity) thrown internally.
                            node = game
                            while node.variations:
                                try:
                                    next_node = node.variation(0)
                                    move = next_node.move
                                    
                                    # Check if the move is in the current board's legal moves list
                                    if move not in board.legal_moves:
                                        is_game_corrupted = True
                                        break
                                        
                                    san_moves.append(board.san(move))
                                    board.push(move)
                                    node = next_node
                                    
                                    if len(san_moves) >= MAX_MOVES - 2:
                                        break
                                except Exception:
                                    # Any move conversion or deduction error will directly mark the current game as damaged.
                                    is_game_corrupted = True
                                    break
                            
                            # A game will only be recorded if the set number of steps has been completed without any illegal markers.
                            if not is_game_corrupted and san_moves:
                                games_san.append(san_moves)
                                local_vocab.update(san_moves)
                            
        print(f"Perfectly completed: {zip_filename} | Final safe extraction games: {len(games_san)}")
        return games_san, local_vocab
    except Exception as e:
        print(f"Error {zip_filename}: {e}")
        return [], set()


def main():
    zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.zip')]
    if not zip_files:
        print(f"Did not find any .zip dataset files in {DATA_DIR}!")
        return

    print(f"Found {len(zip_files)} zip archives, starting multi-core CPU processing pipeline...")
    
    try:
        # Get the actual number of available CPU cores allocated to the current task (e.g., return exactly 16).
        num_workers = len(os.sched_getaffinity(0))
    except AttributeError:
        num_workers = os.cpu_count() or 4
        
    num_workers = min(len(zip_files), num_workers)
    
    all_games = []
    global_vocab = set(SPECIAL_TOKENS)
    
    # Parallel parsing of multiple compressed files using a process pool
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_zip_archive, zip_files)
        
    for games, vocab in results:
        all_games.extend(games)
        global_vocab.update(vocab)
        
    print(f"\nAll PGN parsing completed! Total high-quality games extracted: {len(all_games)}")
    
    # Build vocabulary mapping (Token -> ID)
    # Sort alphabetically to ensure stable ID mappings across runs
    sorted_vocab = SPECIAL_TOKENS + sorted(list(global_vocab - set(SPECIAL_TOKENS)))
    token2id = {token: idx for idx, token in enumerate(sorted_vocab)}
    
    # Save the vocabulary for training and inference loading.
    with open(VOCAB_FILE, 'w', encoding='utf-8') as vf:
        json.dump(token2id, vf, indent=2)
    print(f"Vocabulary built successfully, total Token types: {len(token2id)} | Saved to {VOCAB_FILE}")
    
    # Convert text tokens to integer tensor matrix
    print("Executing tokenization encoding and converting to PyTorch tensors...")
    pad_id = token2id[PAD_TOKEN]
    bos_id = token2id[BOS_TOKEN]
    eos_id = token2id[EOS_TOKEN]
    
    tensor_list = []
    for moves in all_games:
        # Construct a fixed-length sequence of the form: [<s>, e4, e5, Nf3, ..., </s>, <pad>, <pad>].
        move_ids = [bos_id] + [token2id[m] for m in moves] + [eos_id]
        # Fill to a fixed length
        padded_ids = move_ids + [pad_id] * (MAX_MOVES - len(move_ids))
        tensor_list.append(padded_ids)
    
    print("Releasing text cache space...", flush=True)
    del all_games
    gc.collect()
    # Pack them into a contiguous integer Tensor (the number of rows is the number of games, and the number of columns is MAX_MOVES).
    dataset_tensor = torch.tensor(tensor_list, dtype=torch.long)
    
    output_file = os.path.join(OUTPUT_DIR, "chess_dataset.pt")
    torch.save(dataset_tensor, output_file)
    print(f"Preprocessing completed successfully! Final dataset tensor Shape: {dataset_tensor.shape}")
    print(f"Successfully saved to disk: {output_file}")


if __name__ == "__main__":
    main()