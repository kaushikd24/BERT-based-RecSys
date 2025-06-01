import numpy
import math
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

"""""""""

THIS IMPLEMENTS BI-DIRECTIONAL MULTI-HEAD SELF ATTENTION

TO IMPLEMENT BI-DIRECTIONAL SELF ATTENTION WE NEED TO SET MASK = NONE IN THE FORWARD PASS

"""""""""

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


#define bi-directional multi-head self attention by setting mask = None


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
    
    