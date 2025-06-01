import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Emb(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super().__init__()
        self.ItemEmb = nn.Embedding(vocab_size, d_model)
        self.PosEmb = nn.Embedding(max_len, d_model)
        
    def forward(self, tokens, positions):
        input_emb = self.ItemEmb(tokens) + self.PosEmb(positions)
        return input_emb

class ScaledDotProductAttention(nn.Module):
    def __init__ (self):
        super().__init__()
        
    def forward(self, q, k, v, mask):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)
        
        return output, weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, self.d_k * num_heads)
        self.w_k = nn.Linear(d_model, self.d_k*num_heads)
        self.w_v = nn.Linear(d_model, self.d_v*num_heads)
        
        self.attention = ScaledDotProductAttention()
        
        self.w_o = nn.Linear(self.d_v*num_heads, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, -1, self.num_heads, self.d_v).transpose(1,2)
        
        outputs, weights = self.attention(Q, K, V, mask)
        
        outputs = outputs.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        outputs = self.w_o(outputs)
        
        return outputs, weights

class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model, num_heads)
        self.Dropout = nn.Dropout(0.1)
        self.MLP = MLP(d_model, d_ff)
        self.Norm1 = nn.LayerNorm(d_model)
        self.Norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        attn,  _ = self.MHA(x,x,x)
        x = self.Norm1(x + self.Dropout(attn))
        x = self.Norm2(x + self.Dropout(self.MLP(x)))
        return x

class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, max_len, d_model, d_ff, num_heads, num_layers):
        super().__init__()
        self.embedding = Emb(vocab_size, d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            Encoder(d_model, d_ff, num_heads) for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, tokens, positions):
        x = self.embedding(tokens, positions)
        for encoder in self.encoder_layers:
            x = encoder(x)
        x = self.output(x)
        return x 