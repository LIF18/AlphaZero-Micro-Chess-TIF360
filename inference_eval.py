import torch
import json
import random
import chess
import chess.engine
from transformers import GPT2LMHeadModel
from tqdm import tqdm

# --- Configuration Parameters ---
MODEL_PATH = "./chess_model_5M"  # Folder generated after training. If you want to test a semi-finished product, you can use ./chess_model_checkpoints/checkpoint-xxx
EVAL_DATA_PATH = "tokenized_data/dataset_eval.pt"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TEST_POSITIONS = 1000  # Test 1000 different endgame positions
STOCKFISH_DEPTH = 15       # Engine thinking depth (15 is enough for Master level and fast)

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token

def main():
    print("Loading vocabulary, dataset, and large model...")
    token2id, id2token = load_vocab()
    
    # Force using a single GPU for inference. If no GPU is available, fallback to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load pure Transformer model
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Switch to inference mode, disable Dropout
    print(f"The model is loaded to {device}, ready for inference!")

    # Load the test dataset
    eval_data = torch.load(EVAL_DATA_PATH)
    
    # Start the Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    
    top1_correct = 0
    top3_correct = 0
    valid_tests = 0
    
    print(f"Initializing {NUM_TEST_POSITIONS} Master-level Chess Positions for Evaluation...")
    
    # Randomly sample games for testing
    test_indices = random.sample(range(len(eval_data)), NUM_TEST_POSITIONS)
    
    for idx in tqdm(test_indices, desc="Evaluation Progress"):
        sequence = eval_data[idx].tolist()
        
        # Clean out padding tokens
        pad_id = token2id.get("<pad>")
        sequence = [tok for tok in sequence if tok != pad_id]
        
        # At least a few moves are needed to form a middlegame, remove <s> and </s>
        if len(sequence) < 20:
            continue
            
        bos_id = token2id.get("<s>")
        eos_id = token2id.get("</s>")
        
        # Extract pure SAN moves part (remove the leading <s>)
        san_ids = [tok for tok in sequence if tok not in (bos_id, eos_id)]
        
        # Randomly select a truncation point (turn), between opening (10 moves) and endgame
        trunc_point = random.randint(10, len(san_ids) - 5)
        context_ids = san_ids[:trunc_point]
        
        # 1. Restore the true board state
        board = chess.Board()
        is_corrupted = False
        try:
            for move_id in context_ids:
                san_str = id2token[move_id]
                board.push_san(san_str)
        except Exception:
            continue  # If the sampled game has anomalies, skip it
            
        # 2. Summon Stockfish to find the "Absolute Truth"
        try:
            result = engine.play(board, chess.engine.Limit(depth=STOCKFISH_DEPTH))
            stockfish_best_move_san = board.san(result.move)
        except Exception:
            continue
            
        # 3. Let our Transformer predict purely by intuition
        # Construct input: [<s>, move1, move2, ..., trunc_move]
        input_seq = [bos_id] + context_ids
        input_tensor = torch.tensor([input_seq], dtype=torch.long).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_tensor)
            # Extract the logits at the last position of the sequence (i.e., prediction for the next step)
            next_token_logits = outputs.logits[0, -1, :]
            
            # Extract the Top-3 Tokens with the highest probability
            top3_values, top3_indices = torch.topk(next_token_logits, 3)
            
        top3_predicted_sans = [id2token[idx.item()] for idx in top3_indices]
        
        # 4. Calculate Agreement
        if stockfish_best_move_san == top3_predicted_sans[0]:
            top1_correct += 1
        
        if stockfish_best_move_san in top3_predicted_sans:
            top3_correct += 1
            
        valid_tests += 1

    engine.quit()

    # --- Output final report card ---
    print("\n" + "="*40)
    print("🏆 Transformer Chess Intuition Assessment Report 🏆")
    print("="*40)
    print(f"Valid test endgame count : {valid_tests}")
    print(f"Stockfish search depth : Depth {STOCKFISH_DEPTH}")
    print(f"🎯 Top-1 best move agreement : {(top1_correct / valid_tests) * 100:.2f} %")
    print(f"🎯 Top-3 best move agreement : {(top3_correct / valid_tests) * 100:.2f} %")
    print("="*40)
    # print("Note: Stockfish calculated to a depth of 15, while the Transformer outputs with 0 search.")

if __name__ == "__main__":
    main()