import os
import zipfile
import io
import json
from multiprocessing import Pool
import chess.pgn
import torch

DATA_DIR = "lichess_elite_dataset"
OUTPUT_DIR = "tokenized_data"
VOCAB_FILE = "vocab.json"
MAX_MOVES = 128
MIN_ELO = 2400

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]

def process_zip_archive(zip_filename):
    zip_path = os.path.join(DATA_DIR, zip_filename)
    games_san = []
    local_vocab = set()
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                if file_info.filename.endswith('.pgn'):
                    print(f"正在全方位安全解析: {zip_filename} ...")
                    with z.open(file_info) as f:
                        # 忽略所有编解码错误，防止乱码引发中断
                        stream = io.TextIOWrapper(f, encoding='utf-8', errors='ignore')
                        
                        while True:
                            # 1. 捕获读取棋局头部信息时的底层异常
                            try:
                                # 传递纯文本流，遇到格式损坏的棋局直接返回 None 或抛错
                                game = chess.pgn.read_game(stream)
                            except Exception:
                                # 如果当前局头损坏，跳过并尝试继续往下读
                                continue
                                
                            if game is None:
                                break
                            
                            # 2. 检查 Elo 门槛
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
                            
                            # 3. 极其关键的防御：逐步 (Move-by-move) 验证
                            # 原始的 game.mainline_moves() 是一个生成器，如果在中间遇到坏着法会直接崩溃
                            # 我们必须手动遍历它的节点来捕获内部抛出的 ValueError (illegal san)
                            node = game
                            while node.variations:
                                try:
                                    next_node = node.variation(0)
                                    move = next_node.move
                                    
                                    # 检查该走法是否在当前棋盘的合法走法列表中
                                    if move not in board.legal_moves:
                                        is_game_corrupted = True
                                        break
                                        
                                    san_moves.append(board.san(move))
                                    board.push(move)
                                    node = next_node
                                    
                                    if len(san_moves) >= MAX_MOVES - 2:
                                        break
                                except Exception:
                                    # 任何着法转换或推演错误，直接标记当前对局损坏
                                    is_game_corrupted = True
                                    break
                            
                            # 只有当完整走完了设定的步数，且中途没有任何非法标记时，才收录该局
                            if not is_game_corrupted and san_moves:
                                games_san.append(san_moves)
                                local_vocab.update(san_moves)
                                
        print(f"成功跑通归档: {zip_filename} | 安全提取合格对局数: {len(games_san)}")
        return games_san, local_vocab
    except Exception as e:
        print(f"处理压缩包发生不可恢复错误 {zip_filename}: {e}")
        return [], set()

# main 函数无需修改
def main():
    zip_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.zip')]
    if not zip_files:
        return
    num_workers = min(len(zip_files), os.cpu_count() or 4)
    all_games = []
    global_vocab = set(SPECIAL_TOKENS)
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_zip_archive, zip_files)
        
    for games, vocab in results:
        all_games.extend(games)
        global_vocab.update(vocab)
        
    sorted_vocab = SPECIAL_TOKENS + sorted(list(global_vocab - set(SPECIAL_TOKENS)))
    token2id = {token: idx for idx, token in enumerate(sorted_vocab)}
    
    with open(VOCAB_FILE, 'w', encoding='utf-8') as vf:
        json.dump(token2id, vf, indent=2)
        
    pad_id = token2id[PAD_TOKEN]
    bos_id = token2id[BOS_TOKEN]
    eos_id = token2id[EOS_TOKEN]
    
    tensor_list = []
    for moves in all_games:
        move_ids = [bos_id] + [token2id[m] for m in moves] + [eos_id]
        padded_ids = move_ids + [pad_id] * (MAX_MOVES - len(move_ids))
        tensor_list.append(padded_ids)
        
    dataset_tensor = torch.tensor(tensor_list, dtype=torch.long)
    torch.save(dataset_tensor, os.path.join(OUTPUT_DIR, "chess_dataset.pt"))
    print("成功输出干净的数据集张量！")

if __name__ == "__main__":
    main()