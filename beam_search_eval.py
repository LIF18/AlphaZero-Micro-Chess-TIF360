import torch
import json
import random
import chess
import chess.engine
from transformers import GPT2LMHeadModel
from tqdm import tqdm

# --- Core Configuration ---
MODEL_PATH = "./chess_model_5M"
EVAL_DATA_PATH = "tokenized_data/dataset_eval.pt"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TEST_POSITIONS = 500
BEAM_WIDTH = 3           
SEARCH_DEPTH = 3
STOCKFISH_TRUTH_DEPTH = 15

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token

def main():
    print("Loading Pure Transformer Beam Search Engine...")
    token2id, id2token = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    eval_data = torch.load(EVAL_DATA_PATH)
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    
    bos_id = token2id.get("<s>")
    pad_id = token2id.get("<pad>")
    eos_id = token2id.get("</s>")
    
    top1_correct = 0
    valid_tests = 0
    
    test_indices = random.sample(range(len(eval_data)), min(NUM_TEST_POSITIONS, len(eval_data)))
    
    for idx in tqdm(test_indices, desc="Beam Search Evaluation Progress"):
        sequence = eval_data[idx].tolist()
        sequence = [tok for tok in sequence if tok not in (pad_id, bos_id, eos_id)]
        if len(sequence) < 20: continue
            
        trunc_point = random.randint(10, len(sequence) - 5)
        context_ids = sequence[:trunc_point]
        
        # Restore the physical chessboard
        root_board = chess.Board()
        try:
            for move_id in context_ids:
                root_board.push_san(id2token[move_id])
        except Exception:
            continue
            
        try:
            result = engine.play(root_board, chess.engine.Limit(depth=STOCKFISH_TRUTH_DEPTH))
            truth_san = root_board.san(result.move)
        except Exception:
            continue
            
        # Pure Language Model System 2: Beam Search
        # Beam Structure: (Current Token Sequence, Cumulative Log Probability, Corresponding Physical Chessboard, First Move of the Path)
        beams = [( [bos_id] + context_ids, 0.0, root_board.copy(), None )]
        
        for step in range(SEARCH_DEPTH):
            new_beams = []
            
            for seq, cumulative_log_prob, board, first_move in beams:
                if board.is_game_over():
                    new_beams.append((seq, cumulative_log_prob, board, first_move))
                    continue
                    
                input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_tensor).logits[0, -1, :]
                    # Convert to Log probability for easier accumulation.
                    log_probs = torch.log_softmax(logits, dim=-1)
                
                # Take more to prevent encountering illegal steps
                topk_vals, topk_indices = torch.topk(log_probs, BEAM_WIDTH * 2)
                
                added = 0
                for val, token_idx in zip(topk_vals, topk_indices):
                    san_str = id2token.get(token_idx.item())
                    if not san_str: continue
                    
                    try:
                        move = board.parse_san(san_str)
                        if move in board.legal_moves:
                            new_board = board.copy()
                            new_board.push(move)
                            new_seq = seq + [token_idx.item()]
                            new_log_prob = cumulative_log_prob + val.item()
                            
                            # Record the first move of this timeline
                            new_first = san_str if first_move is None else first_move
                            
                            new_beams.append((new_seq, new_log_prob, new_board, new_first))
                            added += 1
                            if added == BEAM_WIDTH: break
                    except ValueError:
                        continue # Filtering large models to prevent illusionary illegal steps
            
            # Sort by cumulative confidence value along the entire path, retaining only the top BEAM_WIDTH parallel universes.
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:BEAM_WIDTH]
            
        
        if not beams: continue
        best_beam_san = beams[0][3]
        
        if best_beam_san == truth_san:
            top1_correct += 1
        valid_tests += 1

    engine.quit()

    print("\n" + "="*45)
    print(" Pure Transformer + Beam Search Report")
    print("="*45)
    print(f"Valid Tests : {valid_tests} positions")
    print(f"LLM Self-Play Depth : {SEARCH_DEPTH} steps (Beam Width: {BEAM_WIDTH})")
    print(f"Absolute Truth Depth : Depth {STOCKFISH_TRUTH_DEPTH}")
    print(f"🚀 Beam Search Top-1 Accuracy : {(top1_correct / valid_tests) * 100:.2f} %")
    print("="*45)

if __name__ == "__main__":
    main()