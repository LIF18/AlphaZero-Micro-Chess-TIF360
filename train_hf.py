import torch
import os
import numpy as np
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments

# 1. Define an extremely minimal PyTorch Dataset wrapper
class ChessDataset(Dataset):
    def __init__(self, pt_file):
        print(f"Loading dataset into memory: {pt_file} ...")
        # Load our split 5 million game records data
        self.data = torch.load(pt_file)
        print(f"Loading successful! Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # For autoregressive models (Causal LM), labels are the input_ids themselves.
        # The model internally automatically shifts input_ids to the right by one position to calculate Next-token Loss.
        return {"input_ids": item, "labels": item.clone()}
    

# 2. Define accuracy calculation function (used to generate evaluation curves)
def compute_metrics(eval_pred):
    # logits, labels = eval_pred
    predictions, labels = eval_pred
    # Get the Token ID with the highest predicted probability for each position
    # predictions = np.argmax(logits, axis=-1)

    labels = labels[:, 1:]  # Because the model internally shifts automatically, we align the labels
    predictions = predictions[:, :-1]  # Similarly, ignore the last predicted value

    # Ignore padding (assuming pad_id is 0, depends specifically on your vocab.json)
    # CausalLM internally shifted automatically, we compare directly
    mask = labels != 0 
    correct = (predictions[mask] == labels[mask]).sum()
    total = mask.sum()
    
    # The returned results will be automatically recorded on the TensorBoard curve
    return {"accuracy": correct / total}


def main():
    # Initialize model architecture (a small powerful engine with about 15M parameters)
    config = GPT2Config(
        vocab_size=14459,      # Replace with the real number of Tokens in your vocab.json
        n_positions=128,      # Context length (MAX_MOVES)
        n_embd=256,           
        n_layer=6,            
        n_head=8              
    )
    model = GPT2LMHeadModel(config)
    print(f"Model initialization completed, parameter count: {model.num_parameters() / 1e6:.2f} M")

    # Prepare dataset
    # Strongly recommended: First run through the entire process with dataset_100k.pt, then switch to dataset_5M.pt
    dataset = ChessDataset("tokenized_data/dataset_5M.pt")
    eval_dataset = ChessDataset("tokenized_data/dataset_eval.pt")

    # Configure industrial-grade training parameters (TrainingArguments is the soul of Hugging Face)
    training_args = TrainingArguments(
        output_dir="./chess_model_checkpoints_5M", # Checkpoint save path
        # overwrite_output_dir=True,
        num_train_epochs=3,                     # Number of training epochs
        per_device_train_batch_size=128,        # Batch Size per GPU (L40S 48GB VRAM is huge, 256 is no pressure at all)

        per_device_eval_batch_size=32,         # During evaluation, Batch Size can be appropriately reduced to save VRAM
        eval_accumulation_steps=10,

        eval_strategy="steps",                  # Periodic evaluation
        eval_steps=1000,                        # Take a test on the validation set every 1000 steps
        logging_steps=100,                      # Print Loss every 100 steps
        report_to="tensorboard",                # Core magic: Enable TensorBoard recording
        logging_dir="./runs/chess_transformer", # Folder where curve data is stored

        save_steps=5000,                        # Save the model every 5000 steps (prevent system kills)
        save_total_limit=3,                     # Keep at most the recent 3 Checkpoints, saving hard drive space
        bf16=True,                              # L40S perfectly supports BF16 mixed precision, doubling speed and extremely stable!
        dataloader_num_workers=4,               # Enable multi-threaded data loading
        # report_to="none"                        # Turn off default wandb monitoring, keep the terminal clean
    )

    # Instantiate the god-level tool Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # One-click start distributed training!
    print("Engine ignited, starting training!")
    trainer.train()
    
    # Save the model weights in their final form
    trainer.save_model("./chess_model_5M")
    print("Training completed successfully, model saved!")

# Before caching prediction results, immediately take argmax to discard the massive probability distribution
def preprocess_logits_for_metrics(logits, labels):
    # The logits returned by HF models are sometimes a tuple, take the first core tensor
    if isinstance(logits, tuple):
        logits = logits[0]
    # Dimensionality reduction directly at this stage! From (batch, 128, 4100) to (batch, 128)
    preds = logits.argmax(dim=-1)
    return preds

if __name__ == "__main__":
    main()