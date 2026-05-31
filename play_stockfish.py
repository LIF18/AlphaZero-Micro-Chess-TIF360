import torch
import json
import chess
import chess.engine
from transformers import GPT2LMHeadModel
from tqdm import tqdm

# --- Core Configuration ---
MODEL_PATH = "./chess_model_5M"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_GAMES = 20           # Total number of evaluation games (run 20 games first to test the waters)
TARGET_ELO = 1350        # The restricted ELO score set for Stockfish (minimum can be set around 1320)
MAX_MOVES = 200          # Maximum half-moves per game to prevent AI from endlessly circling and drawing

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token

def main():
    print(f"⏳ Loading large model brain and Stockfish engine (Target ELO: {TARGET_ELO})...")
    token2id, id2token = load_vocab()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    
    engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    # Core limit: Enable UCI_LimitStrength and suppress engine intelligence to our specified ELO
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": TARGET_ELO})
    
    bos_id = token2id.get("<s>")
    pad_id = token2id.get("<pad>")
    
    # Scoreboard
    results = {"win": 0, "draw": 0, "loss": 0}
    
    print(f"⚔️ AI World Cup begins! Transformer Model vs Stockfish ({TARGET_ELO} Elo) - Total {NUM_GAMES} Games")
    
    for game_idx in tqdm(range(NUM_GAMES), desc="Match Progress"):
        board = chess.Board()
        # Play white one game, play black the next
        model_is_white = (game_idx % 2 == 0)
        
        # Maintain an independent SAN history list
        san_history = []
        
        while not board.is_game_over() and len(board.move_stack) < MAX_MOVES:
            # Determine whose turn it is
            if (board.turn == chess.WHITE) == model_is_white:
                # ==========================================
                # Transformer Large Model Turn (System 1 Intuitive Deduction)
                # ==========================================
                # Restore historical sequence (ensure complete memory for each inference)
                # Get historical Token IDs. If a rare Stockfish move is not in the vocab, use pad to prevent crashing
                sequence = [bos_id] + [token2id.get(san, pad_id) for san in san_history]
                
                # Sliding window truncation! Prevent exceeding the model's maximum context length (128)
                MAX_CONTEXT_LEN = 128
                if len(sequence) > MAX_CONTEXT_LEN:
                    # If exceeded, chop off the earliest moves and only feed the model the most recent 128 moves
                    sequence = sequence[-MAX_CONTEXT_LEN:]
                
                input_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
                
                # Model outputs intuitive probability prediction
                with torch.no_grad():
                    logits = model(input_ids=input_tensor).logits[0, -1, :]
                
                # Core: Legal Move Mask
                best_legal_move = None
                best_legal_val = -float('inf')
                
                # Iterate through all absolutely legal moves on the current physical board
                for move in board.legal_moves:
                    san_str = board.san(move)
                    if san_str in token2id:
                        token_idx = token2id[san_str]
                        logit_val = logits[token_idx].item()
                        
                        # Pick the move with the highest probability from the "legal moves" according to the large model
                        if logit_val > best_legal_val:
                            best_legal_val = logit_val
                            best_legal_move = move
                            
                # If extremely rarely no legal move within the vocab is found, play a random legal move to survive
                if best_legal_move is None:
                    best_legal_move = list(board.legal_moves)[0]
                    
                # Record SAN string and make the move (board.san must be called before push)
                san_history.append(board.san(best_legal_move))
                board.push(best_legal_move)
                
            else:
                # ==========================================
                # Stockfish Engine Turn (Gatekeeper with suppressed ELO)
                # ==========================================
                # Limit calculation time, max limit to the set ELO
                engine_result = engine.play(board, chess.engine.Limit(time=0.1))
                
                # Record SAN string and make the move
                san_history.append(board.san(engine_result.move))
                board.push(engine_result.move)

        # Settle the game result
        outcome = board.result()
        if outcome == "1/2-1/2" or len(board.move_stack) >= MAX_MOVES:
            results["draw"] += 1
        elif (outcome == "1-0" and model_is_white) or (outcome == "0-1" and not model_is_white):
            results["win"] += 1
        else:
            results["loss"] += 1

    engine.quit()

    # --- Output Post-Match Report ---
    print("\n" + "="*45)
    print(f"🏆 Large Model ELO Placement Match Report 🏆")
    print("="*45)
    print(f"Opponent Strength : Stockfish (UCI_Elo = {TARGET_ELO})")
    print(f"Total Games       : {NUM_GAMES} Games")
    print(f"Final Record      : {results['win']} Wins | {results['draw']} Draws | {results['loss']} Losses")
    
    # Win rate calculation (draws count as half a win)
    win_rate = (results['win'] + 0.5 * results['draw']) / NUM_GAMES
    print(f"📊 Overall Win Rate : {win_rate * 100:.2f} %")
    
    if win_rate > 0.5:
        print("💡 Conclusion: Our model's strength has exceeded the engine in this score range! Try raising the TARGET_ELO!")
    else:
        print("💡 Conclusion: This score range is slightly challenging for the current purely intuitive large model.")
    print("="*45)

if __name__ == "__main__":
    main()