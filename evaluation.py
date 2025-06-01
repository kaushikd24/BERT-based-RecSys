import torch
import numpy as np

# Assuming model, eval_loader, and device are defined elsewhere

def evaluate(model, test_loader, k=10):
    model.eval()
    HRs, NDCGs = [], []

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            positions = torch.arange(max_len).unsqueeze(0).expand(tokens.size(0), max_len).to(device)

            # forward pass
            logits = model(tokens, positions)  # [B, L, V]

            # we only care about the final position (where [MASK] was added)
            mask_pos = (tokens == mask_token_id).nonzero(as_tuple=False)  # [N, 2]
            for b in range(tokens.size(0)):
                final_pos = mask_pos[mask_pos[:, 0] == b][:, 1]
                if len(final_pos) == 0:
                    continue
                pos = final_pos[-1].item()
                scores = logits[b, pos]  # [V]
                true_item = labels[b, pos].item()

                top_k = torch.topk(scores, k=k).indices.tolist()
                HRs.append(int(true_item in top_k))

                if true_item in top_k:
                    rank = top_k.index(true_item)
                    NDCGs.append(1 / np.log2(rank + 2))
                else:
                    NDCGs.append(0)

    print(f"HR@{k}: {np.mean(HRs):.4f}, NDCG@{k}: {np.mean(NDCGs):.4f}")
    return np.mean(HRs), np.mean(NDCGs) 