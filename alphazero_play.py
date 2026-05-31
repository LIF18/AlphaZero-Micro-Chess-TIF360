import torch
import torch.nn as nn
import json
import chess
import chess.engine
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import math

# --- Core Configuration ---
ALPHAZERO_MODEL_PATH = "./chess_alphazero_model"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_GAMES = 20           
TARGET_ELO = 1350       # First drop to 1350 Elo to verify the model's true strength before it loses context
MAX_MOVES = 120          # Extremely important! Absolutely cannot exceed 128 moves, otherwise the model's positional encoding will completely collapse.
TOP_K_CANDIDATES = 10    # Give the value network a broader candidate vision.
C_PUCT = 1.5             # Core AlphaZero constant, balancing intuition and value.

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    return token2id, {v: k for k, v in token2id.items()}

class ChessAlphaZeroModel(nn.Module):
    def __init__(self, config_dir):
        super().__init__()
        config = GPT2Config.from_pretrained(config_dir)
        self.gpt = GPT2LMHeadModel(config)
        hidden_size = self.gpt.config.n_embd
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, input_ids):
        outputs = self.gpt(input_ids=input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        policy_logits = outputs.logits[:, -1, :] 
        value_pred = self.value_head(last_hidden_state).squeeze(-1)
        return policy_logits, value_pred

def main():
    print(f"⏳ Waking up the ultimate AlphaZero hybrid model (PUCT-Lite engine)...")
    token2id, id2token = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChessAlphaZeroModel(ALPHAZERO_MODEL_PATH)
    model.load_state_dict(torch.load(f"{ALPHAZERO_MODEL_PATH}/pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()
    
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": TARGET_ELO})
    
    bos_id = token2id.get("<s>")
    pad_id = token2id.get("<pad>")
    results = {"win": 0, "draw": 0, "loss": 0}
    
    print(f"⚔️ The Final Battle! AlphaZero (PUCT heuristic search) vs Stockfish ({TARGET_ELO} Elo)")
    
    for game_idx in tqdm(range(NUM_GAMES), desc="Battle Progress"):
        board = chess.Board()
        model_is_white = (game_idx % 2 == 0)
        san_history = []
        
        while not board.is_game_over() and len(board.move_stack) < MAX_MOVES:
            if (board.turn == chess.WHITE) == model_is_white:
                sequence = [bos_id] + [token2id.get(san, pad_id) for san in san_history]
                
                # Forced risk avoidance: If it really exceeds the limit, it's better to let the engine move randomly than let the model go crazy and collapse.
                if len(sequence) >= 127:
                    board.push(list(board.legal_moves)[0])
                    continue
                
                input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
                
                with torch.no_grad():
                    logits, _ = model(input_tensor)
                    logits = logits[0]
                    
                # Calculate the true Softmax probability of legal moves
                legal_moves_data = []
                legal_logits = []
                for move in board.legal_moves:
                    san_str = board.san(move)
                    if san_str in token2id:
                        legal_moves_data.append((move, san_str))
                        legal_logits.append(logits[token2id[san_str]].item())
                
                if not legal_moves_data:
                    board.push(list(board.legal_moves)[0])
                    continue
                    
                # Convert Logits to probabilities (0.0 ~ 1.0)
                legal_probs = torch.softmax(torch.tensor(legal_logits), dim=0).tolist()
                
                # Extract Top-K
                candidates = list(zip(legal_moves_data, legal_probs))
                candidates.sort(key=lambda x: x[1], reverse=True)
                top_candidates = candidates[:TOP_K_CANDIDATES]
                
                best_move = None
                best_combined_score = -float('inf')
                
                for (move, san_str), prior_prob in top_candidates:
                    candidate_seq = sequence + [token2id[san_str]]
                    cand_tensor = torch.tensor([candidate_seq], dtype=torch.long).to(device)
                    
                    with torch.no_grad():
                        _, val_pred = model(cand_tensor)
                        val = val_pred.item()
                        
                    # Negating the opponent's win rate gives my positional advantage
                    my_value = -val
                    
                    # Core Magic: PUCT scoring formula Positional advantage + constant * intuitive probability
                    combined_score = my_value + C_PUCT * prior_prob
                    
                    if combined_score > best_combined_score:
                        best_combined_score = combined_score
                        best_move = move
                            
                san_history.append(board.san(best_move))
                board.push(best_move)
            else:
                # Stockfish turn
                engine_result = engine.play(board, chess.engine.Limit(time=0.1))
                san_history.append(board.san(engine_result.move))
                board.push(engine_result.move)

        outcome = board.result()
        if outcome == "1/2-1/2" or len(board.move_stack) >= MAX_MOVES:
            results["draw"] += 1
        elif (outcome == "1-0" and model_is_white) or (outcome == "0-1" and not model_is_white):
            results["win"] += 1
        else:
            results["loss"] += 1

    engine.quit()

    print("\n" + "="*45)
    print(f"🏆 AlphaZero Actual Combat Evaluation Report 🏆")
    print(f"Opponent : Stockfish ({TARGET_ELO} Elo)")
    print(f"Record   : {results['win']} Wins | {results['draw']} Draws | {results['loss']} Losses")
    win_rate = (results['win'] + 0.5 * results['draw']) / NUM_GAMES
    print(f"📈 Overall Win Rate : {win_rate * 100:.2f} %")
    print("="*45)

if __name__ == "__main__":
    main()