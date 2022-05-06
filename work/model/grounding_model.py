# -*- coding: utf-8 -*-
from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.visual_backbone import build_visual_backbone
from .backbone.rnn import build_textual_encoder
from .vgtr.vgtr import build_vgtr

from .backbone.early_attention import EarlyDotAttention, EarlyCoAttention
from .vgtr.position_encoding import PositionEmbeddingSine, PositionEncoding1D

class GroundingModel(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.visual_encoder = build_visual_backbone(args)
        self.textual_encoder = build_textual_encoder(args)
        self.vgtr = build_vgtr(args)
        self.num_exp_tokens = args.num_exp_tokens
        self.prediction_head = nn.Sequential(
            nn.Linear(args.hidden_dim * args.num_exp_tokens, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.BatchNorm1d(args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, 4)
        )

        self.use_co_attention = args.use_co_attention
        self.pooled_feats_linear = nn.Linear(2048, 256) 
        self.exp_feats_linear = nn.Linear(1024, 256)

        self.pos_encoder = PositionEmbeddingSine(args.hidden_dim//2, normalize=False)
        self.pos_encoder_1d = PositionEncoding1D(256)
        self.pointwise = nn.Conv2d(128, 256, kernel_size=1, stride=1)

        self.early_dot_attn = EarlyDotAttention(l_norm=False)
        self.early_co_attn = EarlyCoAttention()
        
    def forward(self, img, expression_word_id):

        img_feature, pooled_features = self.visual_encoder(img) # pooled feats - B, 128, 4, 4
        exp_feature = self.textual_encoder(expression_word_id) # B, 4, 256
        
        img_exp_feature = []

        # increasing number of channels
        x_high_channels = []
        for feat in pooled_features:
            x_high_channels.append(self.pointwise(feat)) # B, 256, 4, 4

        # finding positional encoding
        pos_embed = []
        for feat in x_high_channels:
            pos_encode = self.pos_encoder(feat)
            pos_embed.append(pos_encode)

        # calculating cross-attention
        for i, feat in enumerate(x_high_channels):           
        
            feat = feat + pos_embed[i] # early addition.  
            x = feat.flatten(2) # B, 256, 16

            x = torch.transpose(x, 1, 2) # B, 16, 256
            if self.use_co_attention:
                img_exp_feature.append(self.early_co_attn(x, exp_feature)) # [(B, 4, 256), ... ]
            else:
                img_exp_feature.append(self.early_dot_attn(x, exp_feature)) # [(B, 4, 256), ... ]

        img_exp_feature = torch.cat(img_exp_feature, dim=1) # B, 16, 256
        img_exp_feature = torch.sigmoid(img_exp_feature)
        pos_embed = torch.stack(pos_embed, dim=3)
        
        embed = self.vgtr(img_exp_feature, exp_feature, None, expression_word_id)
        embed2 = torch.cat([embed[:, i] for i in range(self.num_exp_tokens)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        return pred
