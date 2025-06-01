import torch
from torch.utils.data import DataLoader
import pandas as pd

from model_architecture import BERT4Rec
from training import *
from evaluation import evaluate
from bert4rec_dataset import BERT4RecDataset, BERT4RecEvalDataset

# Load data
user_sequences = pd.read_pickle("data/train_remapped.pkl")

# Initialize data loaders
train_dataset = BERT4RecDataset(user_sequences, max_len=50, mask_prob=0.15, pad_token=0, mask_token_id=3707)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

eval_dataset = BERT4RecEvalDataset(user_sequences, max_len=50, mask_token_id=3707)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

# Initialize model
vocab_size = 3708  # Example vocab size
d_model = 128
d_ff = 4 * d_model
num_heads = 2
num_layers = 2
max_len = 50

model = BERT4Rec(vocab_size, max_len, d_model, d_ff, num_heads, num_layers)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Train the model
loader = train_loader  # Alias for compatibility with training.py

# Assuming train_model is defined in training.py
# train_model(model, loader, device, vocab_size, max_len)

# Evaluate the model
evaluate(model, eval_loader, k=10) 