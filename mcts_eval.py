import torch
import json
import random
import math
import chess
import chess.engine
from transformers import GPT2LMHeadModel
from tqdm import tqdm

# --- Configuration Parameters ---
MODEL_PATH = "./chess_model_5M"
EVAL_DATA_PATH = "tokenized_data/dataset_eval.pt"
VOCAB_PATH = "vocab.json"
ENGINE_PATH = "./stockfish_engine/stockfish/stockfish-ubuntu-x86-64-avx2"

NUM_TEST_POSITIONS = 200     # MCTS is slower, let's test 200 endgames first
STOCKFISH_EVAL_DEPTH = 4    # Depth used to provide the "Absolute Truth"
MCTS_SIMULATIONS = 20        # Number of times MCTS simulates in its 'mind' before each prediction (since there's no batching, 30 times already shows significant improvement)
C_PUCT = 0.5                 # Exploration constant, balancing intuition (Policy) and deep logic (Value)
TOP_K_PRUNING = 3

# --- Utility Functions ---
def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        token2id = json.load(f)
    id2token = {v: k for k, v in token2id.items()}
    return token2id, id2token

# Win rate conversion: Convert Stockfish's Centipawn to a win rate between -1 and 1
def cp_to_win_prob(cp_score):
    if cp_score is None:
        return 0.0
    # Classic chess win rate conversion formula (Sigmoid variant)
    return 2.0 / (1.0 + math.exp(-0.00368208 * cp_score)) - 1.0

# --- MCTS Core Node Class ---
class MCTSNode:
    def __init__(self, board, sequence, parent=None, move_san=None, prior_prob=0.0):
        self.board = board                 # Current node's physical board
        self.sequence = sequence           # Token sequence up to this point (for LLM input)
        self.parent = parent
        self.move_san = move_san           # The move that led to this node
        
        self.children = {}                 # Set of child nodes
        self.visits = 0                    # Visit count N
        self.value_sum = 0.0               # Accumulated value W
        self.prior_prob = prior_prob       # Prior probability P given by the language model
        
    def is_expanded(self):
        return len(self.children) > 0
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def get_ucb(self, c_puct):
        # AlphaZero's PUCT formula
        # U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        q_value = self.value()
        u_value = c_puct * self.prior_prob * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value


def main():
    print(" Loading hybrid AI engine (Transformer Policy + Stockfish Value)...")
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
    
    print(f" Launch MCTS Hybrid Search Evaluation ( {len(test_indices)} matches)...")
    
    for idx in tqdm(test_indices, desc="MCTS Evaluation Progress"):
        sequence = eval_data[idx].tolist()
        sequence = [tok for tok in sequence if tok not in (pad_id, bos_id, eos_id)]
        if len(sequence) < 20: continue
            
        trunc_point = random.randint(10, len(sequence) - 5)
        context_ids = sequence[:trunc_point]
        
        # Restore the true board
        root_board = chess.Board()
        try:
            for move_id in context_ids:
                root_board.push_san(id2token[move_id])
        except Exception:
            continue
            
        # 1. Engine calculates absolute truth
        try:
            result = engine.play(root_board, chess.engine.Limit(depth=STOCKFISH_EVAL_DEPTH))
            truth_san = root_board.san(result.move)
        except Exception:
            continue
            
        # ==========================================
        # 2. MCTS search core loop (Monte Carlo Tree self-simulation)
        # ==========================================
        root_seq = [bos_id] + context_ids
        root = MCTSNode(root_board, root_seq)
        
        for _ in range(MCTS_SIMULATIONS):
            node = root
            
            # --- Phase 1: Selection (Select the most exploratory child node) ---
            while node.is_expanded():
                # Find the child node with the highest UCB score
                best_action = max(node.children.keys(), key=lambda a: node.children[a].get_ucb(C_PUCT))
                node = node.children[best_action]
                
            # --- Phase 2: Expansion (Language model provides intuitive policy + Top-K pruning) ---
            if not node.board.is_game_over():
                input_tensor = torch.tensor([node.sequence], dtype=torch.long).to(device)
                with torch.no_grad():
                    logits = model(input_ids=input_tensor).logits[0, -1, :]
                    
                # [Core Magic]: Directly take Top-K of Logits, obliterate the rest!
                topk_values, topk_indices = torch.topk(logits, TOP_K_PRUNING)
                
                legal_policy = {}
                # Only pick legal moves within Top-K
                for val, idx in zip(topk_values, topk_indices):
                    san_str = id2token.get(idx.item())
                    if san_str is None: continue
                    
                    try:
                        # Try to convert the predicted SAN to a move, it will raise an error if illegal
                        move = node.board.parse_san(san_str)
                        if move in node.board.legal_moves:
                            # Restore to the true probability distribution
                            legal_policy[san_str] = math.exp(val.item()) 
                    except ValueError:
                        continue # The large model hallucinated at this step, proposed an illegal move, throw it away directly
                        
                # Normalize local probabilities
                sum_p = sum(legal_policy.values())
                if sum_p > 0:
                    for san, p in legal_policy.items():
                        child_board = node.board.copy()
                        child_board.push_san(san)
                        child_seq = node.sequence + [token2id[san]]
                        # Create elite child nodes highly filtered by the large model
                        node.children[san] = MCTSNode(child_board, child_seq, parent=node, move_san=san, prior_prob=p/sum_p)
            
            # --- Phase 3: Evaluation (Ultra-fast shallow evaluation acts as the value network) ---
            if node.board.is_game_over():
                outcome = node.board.result()
                if outcome == '1-0': v = 1.0 if node.board.turn == chess.BLACK else -1.0 # White won the previous move
                elif outcome == '0-1': v = -1.0 if node.board.turn == chess.BLACK else 1.0
                else: v = 0.0
            else:
                # Spend only a few milliseconds to let Stockfish look 1 step ahead and give a positional score
                info = engine.analyse(node.board, chess.engine.Limit(depth=1))
                score = info["score"].white() if node.board.turn == chess.WHITE else info["score"].black()
                v = cp_to_win_prob(score.score(mate_score=10000))
                
            # --- Phase 4: Backpropagation ---
            curr = node
            while curr is not None:
                curr.visits += 1
                curr.value_sum += v
                # Perspective flip: Parent node's return is the negative of the current node's return (zero-sum game)
                v = -v
                curr = curr.parent

        # 3. Search ends, extract the best move (by highest visit count, this is MCTS's most robust extraction method)
        if not root.children:
            continue
        mcts_best_san = max(root.children.keys(), key=lambda a: root.children[a].visits)
        
        # 4. Calculate alignment degree
        if mcts_best_san == truth_san:
            top1_correct += 1
        valid_tests += 1

    engine.quit()

    print("\n" + "="*45)
    print("Transformer + MCTS Hybrid Intuition Engine Report")
    print("="*45)
    print(f"Valid evaluation endgames : {valid_tests} games")
    print(f"MCTS single-step simulations : {MCTS_SIMULATIONS} simulations")
    print(f"Absolute truth depth : Depth {STOCKFISH_EVAL_DEPTH}")
    print(f"🚀 MCTS Top-1 best move agreement : {(top1_correct / valid_tests) * 100:.2f} %")
    print("="*45)

if __name__ == "__main__":
    main()