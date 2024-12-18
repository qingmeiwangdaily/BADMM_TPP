import torch.nn as nn

from transformer.SubLayers import MultiHeadAttention_2, PositionwiseFeedForward,MultiHeadAttention_softmax


class EncoderLayer_2(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, n_it, mode, rho, lambda_, alpha, dropout=0.1, normalize_before=True):
        super(EncoderLayer_2, self).__init__()
        self.n_it=n_it
        self.mode=mode
        self.slf_attn = MultiHeadAttention_2(
            n_head, d_model, d_k, d_v, n_it, mode, rho, lambda_, alpha, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
    
class EncoderLayer_softmax(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer_softmax, self).__init__()
        self.slf_attn = MultiHeadAttention_softmax(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
