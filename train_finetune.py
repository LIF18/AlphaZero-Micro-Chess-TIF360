import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import json

# --- Configuration Parameters ---
BASE_MODEL_PATH = "./chess_model_5M"
DISTILLED_DATA_PATH = "tokenized_data/dataset_distilled.pt"
OUTPUT_DIR = "./chess_alphazero_model_2M"
VOCAB_PATH = "vocab.json"
MAX_SEQ_LEN = 128

def load_vocab():
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# Distilled dataset wrapper
class DistilledDataset(Dataset):
    def __init__(self, pt_file, vocab):
        print(f"📦 Loading distilled dataset: {pt_file} ...")
        self.data = torch.load(pt_file)
        self.vocab_size = len(vocab)
        self.pad_id = vocab.get("<pad>", 0)
        print(f"✅ Loading successful! A total of {len(self.data)} high-quality special training samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        seq = item["input_ids"]
        
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[-MAX_SEQ_LEN:]
        else:
            seq = seq + [self.pad_id] * (MAX_SEQ_LEN - len(seq))
            
        input_ids = torch.tensor(seq, dtype=torch.long)
        value_target = torch.tensor(item["value_target"], dtype=torch.float)
        
        policy_targets = torch.zeros(self.vocab_size, dtype=torch.float)
        for move_id_str, prob in item["policy_targets"].items():
            policy_targets[int(move_id_str)] = prob
            
        return {
            "input_ids": input_ids,
            "value_target": value_target,
            "policy_targets": policy_targets
        }

# AlphaZero hybrid network architecture
class ChessAlphaZeroModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        print("🧠 Performing model architecture surgery, mounting Value Head...")
        self.gpt = GPT2LMHeadModel.from_pretrained(base_model_path)
        hidden_size = self.gpt.config.n_embd
        
        # Freeze Transformer base layers
        for param in self.gpt.transformer.parameters():
            param.requires_grad = False
            
        for param in self.gpt.lm_head.parameters():
            param.requires_grad = True
            
        # Value evaluation network
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, input_ids, value_target=None, policy_targets=None, **kwargs):
        outputs = self.gpt(input_ids=input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1][:, -1, :] 
        
        policy_logits = outputs.logits[:, -1, :]
        value_pred = self.value_head(last_hidden_state).squeeze(-1)
        
        loss = None
        if value_target is not None and policy_targets is not None:
            value_loss = F.mse_loss(value_pred, value_target)
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = torch.sum(-policy_targets * log_probs, dim=-1).mean()
            loss = policy_loss + value_loss
            
        return {"loss": loss, "logits": policy_logits, "value_pred": value_pred}

def main():
    vocab = load_vocab()
    model = ChessAlphaZeroModel(BASE_MODEL_PATH)
    
    # Load the complete distilled data and split out the Eval validation set (95% training, 5% exam)
    full_dataset = DistilledDataset(DISTILLED_DATA_PATH, vocab)
    train_size = int(0.95 * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    print(f"🗂️ Dataset split completed: Training set {train_size} games, Validation set {eval_size} games.")
    
    # Define fine-tuning parameters
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,                     
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        
        # Use old version parameter names
        eval_strategy="steps",
        eval_steps=200,                         
        
        # Since the old version does not support disabling safetensors, we directly turn off the intermediate save function!
        save_strategy="no",                     # Do not save checkpoint during training
        # save_steps=1000,                      
        # save_safetensors=False,               
        
        logging_steps=50,
        bf16=True,                              
        dataloader_num_workers=4,
        remove_unused_columns=False,            
        report_to="tensorboard",
        logging_dir="./runs/alphazero_finetune"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset # [New] Mount validation set
    )

    print("🚀 AlphaZero architecture fine-tuning ignited!")
    trainer.train()
    
    # Force using native PyTorch method to save the final hybrid weights
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/pytorch_model.bin")
    # Also save a copy of the GPT model config for easy loading later
    model.gpt.config.save_pretrained(OUTPUT_DIR)
    
    print("🎉 The ultimate divine weapon has been forged! Model and configuration safely saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()