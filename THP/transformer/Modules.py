import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.sinkhorn import SinkhornDistance
from transformer.BADMM import BregmanADMM
from transformer.BADMM12 import BregmanADMM12


class ScaledDotProductAttention_2(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, n_it, mode, rho, lambda_, alpha, temperature, attn_dropout=0.2):
        super().__init__()
        self.n_it = n_it
        self.mode = mode
        self.rho = rho
        self.lambda_ = lambda_
        self.alpha = alpha
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  
        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)  
        
        if self.mode=='softmax':
            attn_weights = self.dropout(F.softmax(attn, dim=-1))
        elif self.mode=='badmm':
            attn = F.softmax(attn, dim=-1)
            attn_shape = attn.shape   # batch_size * n_heads * L*L
            attn = attn.view(-1, attn_shape[2], attn_shape[3]) 
            BADMM = BregmanADMM(self.rho, self.lambda_, self.alpha, self.n_it)
            attn_weights = BADMM(attn)
            attn_weights = attn_weights.view(attn_shape) 
            attn_weights = self.dropout(attn_weights)
        elif self.mode=='badmm12':
            attn = F.softmax(attn, dim=-1)
            attn_shape = attn.shape   # batch_size * n_heads * L*L
            attn = attn.view(-1, attn_shape[2], attn_shape[3]) 
            BADMM = BregmanADMM12(self.rho, self.lambda_, self.alpha, self.n_it)
            attn_weights = BADMM(attn)
            attn_weights = attn_weights.view(attn_shape) 
            attn_weights = self.dropout(attn_weights)
        elif self.mode=='sinkhorn':
            # attn = F.softmax(attn, dim=-1)  
            attn_shape = attn.shape   # batch_size * n_heads * L*L
            attn = attn.view(-1, attn_shape[2], attn_shape[3]) 
            sink = SinkhornDistance(1, max_iter=self.n_it)
            attn_weights = sink(attn)[0]
            attn_weights = attn_weights * attn_shape[-1]
            attn_weights = attn_weights.view(attn_shape) 
            attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class ScaledDotProductAttention_softmax(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn
