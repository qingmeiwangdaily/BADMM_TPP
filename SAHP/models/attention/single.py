import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

from .sinkhorn import SinkhornDistance
from .BADMM import BregmanADMM
from .BADMM12 import BregmanADMM12
from .sinkhorn import SinkhornDistance


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, rho, lambda_, alpha, n_it, mode):
        super().__init__()
        self.n_it = n_it
        self.rho = rho
        self.lambda_ = lambda_
        self.alpha = alpha
        self.mode = mode

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        if self.mode == 'softmax':
            p_attn = F.softmax(scores, dim=-1)
        elif self.mode == 'badmm':
            attn = scores
            attn = F.softmax(attn, dim=-1)
            attn_shape = attn.shape
            attn = attn.view(-1, attn_shape[2], attn_shape[3])
            BADMM = BregmanADMM(self.rho, self.lambda_, self.alpha, self.n_it)
            p_attn = BADMM(attn)
            p_attn = p_attn.view(attn_shape)
        elif self.mode == 'badmm12':
            attn = scores
            attn = F.softmax(attn, dim=-1) 
            attn_shape = attn.shape
            attn = attn.view(-1, attn_shape[2], attn_shape[3])
            BADMM = BregmanADMM12(self.rho, self.lambda_, self.alpha, self.n_it)
            p_attn = BADMM(attn)
            p_attn = p_attn.view(attn_shape)
        elif self.mode == 'sinkhorn':
            attn = scores
            # attn = F.softmax(attn, dim=-1) 
            attn_shape = attn.shape
            attn = attn.view(-1, attn_shape[2], attn_shape[3])
            sink = SinkhornDistance(1, max_iter=self.n_it)
            p_attn = sink(attn)[0]
            p_attn = p_attn * p_attn.shape[-1]
            p_attn = p_attn.view(attn_shape)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

