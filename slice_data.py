import torch
import os

INPUT_FILE = "tokenized_data/chess_dataset.pt"
DIR = "tokenized_data"

print("Loading a massive 21GB tensor into memory (this may take several minutes)...")
full_data = torch.load(INPUT_FILE)

print(f"Loading successful! Original data dimensions: {full_data.shape}")

# First phase of partitioning: 100,000 disks (for debugging)
slice_100k = full_data[:100000]
torch.save(slice_100k, os.path.join(DIR, "dataset_100k.pt"))
print(f"Loading successful! Generated 100k samples -> {slice_100k.shape}")

# Second phase of partitioning: 5,000,000 disks (for core training)
slice_5m = full_data[:5000000]
torch.save(slice_5m, os.path.join(DIR, "dataset_5M.pt"))
print(f"Loading successful! Generated 5M samples -> {slice_5m.shape}")

# 10,000 disks were specifically reserved as a verification set for the exam.
eval_data = full_data[5000000:5010000] 
torch.save(eval_data, os.path.join(DIR, "dataset_eval.pt"))