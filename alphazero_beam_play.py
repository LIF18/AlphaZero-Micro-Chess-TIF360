import torch
import torch.nn as nn
import json
import chess
import chess.engine
from transformers import GPT2LMHeadModel, GPT2Config
from tqdm import tqdm
import math

# --- 核心配置 ---
ALPHAZERO_MODEL_PATH = "./chess_alphazero_model_2M"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_GAMES = 20           
TARGET_ELO = 1350      
MAX_MOVES = 120          # Strictly limit the maximum number of steps to prevent amnesia.
BEAM_WIDTH = 3           # Maintain 3 optimal timelines
SEARCH_DEPTH = 3         # Look forward 3 steps
C_PUCT = 0.5             # Fusion coefficient, balancing intuitive probability and endgame value

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
    print(f"The ultimate AlphaZero hybrid model (Depth-{SEARCH_DEPTH} beam search engine is being unleashed.)...")
    token2id, id2token = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChessAlphaZeroModel(ALPHAZERO_MODEL_PATH)
    model.load_state_dict(torch.load(f"{ALPHAZERO_MODEL_PATH}/pytorch_model.bin", map_location=device))
    model.to(device)
    model.eval()
    
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": TARGET_ELO})
    # engine.configure({"Skill Level": 0})
    
    bos_id = token2id.get("<s>")
    pad_id = token2id.get("<pad>")
    results = {"win": 0, "draw": 0, "loss": 0}
    
    print(f"⚔️ AlphaZero + Beam Search vs Stockfish ({TARGET_ELO})")
    
    for game_idx in tqdm(range(NUM_GAMES), desc="Final battle progress"):
        board = chess.Board()
        model_is_white = (game_idx % 2 == 0)
        san_history = []
        
        while not board.is_game_over() and len(board.move_stack) < MAX_MOVES:
            if (board.turn == chess.WHITE) == model_is_white:
                sequence = [bos_id] + [token2id.get(san, pad_id) for san in san_history]
                
                if len(sequence) >= 127:
                    board.push(list(board.legal_moves)[0])
                    continue
                
                # AlphaZero System 2: Beam Search
                # Structure: (Token sequence, physical chessboard, first step of this timeline, cumulative log probability)
                beams = [(sequence, board.copy(), None, 0.0)]
                
                for step in range(SEARCH_DEPTH):
                    new_beams = []
                    for seq, b, first_move, cum_log_prob in beams:
                        if b.is_game_over():
                            new_beams.append((seq, b, first_move, cum_log_prob))
                            continue

                        input_tensor = torch.tensor([seq[-128:]], dtype=torch.long).to(device)
                        with torch.no_grad():
                            logits, _ = model(input_tensor)
                            logits = logits[0]

                        # Filter valid steps and calculate local Log Prob
                        legal_moves_data = []
                        for move in b.legal_moves:
                            san_str = b.san(move)
                            if san_str in token2id:
                                legal_moves_data.append((move, san_str, logits[token2id[san_str]].item()))

                        if not legal_moves_data: continue

                        # Log Softmax
                        legal_logits = torch.tensor([x[2] for x in legal_moves_data])
                        log_probs = torch.log_softmax(legal_logits, dim=0).tolist()

                        for i, (move, san_str, _) in enumerate(legal_moves_data):
                            new_b = b.copy()
                            new_b.push(move)
                            new_seq = seq + [token2id[san_str]]
                            # If this is the first level of the search, record this move; otherwise, inherit the previous starting move.
                            new_first = move if first_move is None else first_move
                            new_cum_log_prob = cum_log_prob + log_probs[i]

                            new_beams.append((new_seq, new_b, new_first, new_cum_log_prob))

                    # Survival of the fittest: Retain the top B timelines based on intuition and confidence to prevent exponential growth.
                    new_beams.sort(key=lambda x: x[3], reverse=True)
                    beams = new_beams[:BEAM_WIDTH]

                # Value Head
                best_move = None
                best_score = -float('inf')

                if not beams:
                    best_move = list(board.legal_moves)[0]
                else:
                    for seq, b, first_move, cum_log_prob in beams:
                        if b.is_game_over():
                            outcome = b.result()
                            if outcome == "1/2-1/2": val = 0.0
                            elif (outcome == "1-0" and model_is_white) or (outcome == "0-1" and not model_is_white): val = 1.0
                            else: val = -1.0
                            my_value = val
                        else:
                            input_tensor = torch.tensor([seq[-128:]], dtype=torch.long).to(device)
                            with torch.no_grad():
                                _, val_pred = model(input_tensor)
                                val = val_pred.item()
                            
                            # View Alignment: If it's still my turn when we reach a leaf node, val is my win rate;
                            # If it's the opponent's turn, val is their win rate, which I need to invert (-val).
                            leaf_is_white = (b.turn == chess.WHITE)
                            if model_is_white == leaf_is_white:
                                my_value = val
                            else:
                                my_value = -val

                        # Final score: Market value after these 3 steps + average confidence of the intuition network in this path.
                        # math.exp(cum_log_prob / SEARCH_DEPTH) Restore the path log_prob to the average single-step probability (0~1).
                        avg_path_prob = math.exp(cum_log_prob / SEARCH_DEPTH)
                        score = my_value + C_PUCT * avg_path_prob
                        
                        if score > best_score:
                            best_score = score
                            best_move = first_move

                san_history.append(board.san(best_move))
                board.push(best_move)
            else:
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
    print(f"🏆 AlphaZero Beam Search Evaluation Report (Depth-{SEARCH_DEPTH}) 🏆")
    print(f"Opponent : Stockfish ({TARGET_ELO}分)")
    print(f"Results : {results['win']} win | {results['draw']} draw | {results['loss']} loss")
    win_rate = (results['win'] + 0.5 * results['draw']) / NUM_GAMES
    print(f"📈 Overall Win Rate : {win_rate * 100:.2f} %")
    print("="*45)

if __name__ == "__main__":
    main()