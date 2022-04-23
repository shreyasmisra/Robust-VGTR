import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np

class VGEncoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()

        self.hidden_dim = d_model
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = EncoderLayer1(d_model, nhead, dim_feedforward, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        encoder_norm2 = nn.LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm, encoder_norm2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_exp_fused_feat, pos_feature, expression_feature, word_id=None, exp_pos_feature=None):

        src = img_exp_fused_feat.permute(1, 0, 2)  # (4, B, 256)
        pos_embed = pos_feature.permute(1, 0, 2)

        out, expf = self.encoder(src, expression_feature, pos=pos_embed, exp_pos_feature=exp_pos_feature)
        out = out.transpose(0, 1)

        return out, expf


class Encoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, norm2=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.norm2 = norm2

    def forward(self, src,
                expression_feature,
                pos: Optional[Tensor] = None,
                exp_pos_feature=None):

        output = src
        exp_f = expression_feature

        for layer in self.layers:
            output, exp_f = layer(output, exp_f, pos=pos, exp_pos_feature=exp_pos_feature)
        if self.norm is not None:
            output = self.norm(output)
            exp_f = self.norm2(exp_f)

        return output, exp_f


class EncoderLayer1(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.exp_self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.exp_self_norm1 = nn.LayerNorm(d_model)
        self.exp_self_norm2 = nn.LayerNorm(d_model)
        self.exp_self_dropout1 = nn.Dropout(dropout)
        self.expression_ffn_linear1 = nn.Linear(in_features=d_model, out_features=dim_feedforward)
        self.expression_ffn_dropout = nn.Dropout(dropout)
        self.expression_ffn_linear2 = nn.Linear(dim_feedforward, d_model)
        self.expression_ffn_dropout2 = nn.Dropout(dropout)
        self.expression_ffn_activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     expression_feature,
                     pos: Optional[Tensor] = None,
                     exp_pos_feature=None):

        # self-attn for exp feature
        expression_feature = expression_feature.permute(1, 0, 2)
        expression_feature2 = self.exp_self_norm1(expression_feature)
        exp_q = exp_k = self.with_pos_embed(expression_feature2, exp_pos_feature)
        expression_feature2 = self.exp_self_attn(exp_q, exp_k, value=expression_feature2)[0]  # (maxL, bs, d)
        expression_feature = expression_feature + self.exp_self_dropout1(expression_feature2)
        expression_feature = self.exp_self_norm2(expression_feature)
        expression_feature = expression_feature.permute(1, 0, 2)

        expression_feature = expression_feature  # (bs, maxL, d) # use this as grounding queries

        # self-attn for img-text fused feature
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)  # q: (hw, bs, d) (4, B, 256)

        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        fused_vis_text_feature = src2
        fused_expression_feature = expression_feature  # (bs, maxL, d)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(fused_vis_text_feature))))
        src = fused_vis_text_feature + self.dropout2(src2) # output of the visual text fused part of the encoder

        # FFN
        expression_feature2 = self.expression_ffn_linear2(self.expression_ffn_dropout(
            self.expression_ffn_activation(self.expression_ffn_linear1(fused_expression_feature))))
        expression_feature = fused_expression_feature + self.expression_ffn_dropout2(expression_feature2) # grounding queries

        return src, expression_feature

    # self.forward(src, expression_feature, origin_h, origin_w,
    #                      word_mask, src_mask, src_key_padding_mask, pos, exp_pos_feature)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")