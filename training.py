import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Assuming model, loader, and device are defined elsewhere

# Step 2: Define loss & optimizer
criterion = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padded/real tokens
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Step 3: Training loop
epochs = 100

for epoch in range(epochs):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for tokens, labels in loop:
        tokens, labels = tokens.to(device), labels.to(device)
        positions = torch.arange(max_len).unsqueeze(0).expand(tokens.size(0), max_len).to(device)
        
        optimizer.zero_grad()

        logits = model(tokens, positions)           # [B, L, V]
        logits = logits.view(-1, vocab_size)        # [B*L, V]
        labels = labels.view(-1)                    # [B*L]

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}") 