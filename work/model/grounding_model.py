# -*- coding: utf-8 -*-
from multiprocessing import pool
import torch
import torch.nn as nn
from .backbone.visual_backbone import build_visual_backbone
from .backbone.rnn import build_textual_encoder
from .vgtr.vgtr import build_vgtr

from .backbone.early_attention import DotAttention

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

        self.pooled_feats_linear = nn.Linear(2048, 256) # make part of args
        self.exp_feats_linear = nn.Linear(1024, 256)

        self.downsample = nn.Conv2d(128, 256, kernel_size=7, stride=2)

        self.early_attn = DotAttention()

    def forward(self, img, expression_word_id):

        img_feature, pooled_features = self.visual_encoder(img) # pooled feats - B, 128, 16, 16
        exp_feature = self.textual_encoder(expression_word_id) # B, 4, 256
        
        # img_projected_feats = [] # each element is 48, 256
        # for i in range(len(pooled_features)):
        #     pooled_feats_flatten = pooled_features[i].flatten(2).flatten(1)
        #     img_projected_feats.append(self.pooled_feats_linear(pooled_feats_flatten))

        # exp_projected_feats = self.exp_feats_linear(exp_feature.flatten(1)) # 48, 256

        pooled_features = self.downsample(pooled_features) # B, 256, 5, 5
        pooled_features = pooled_features.flatten(2) # B, 256, 25
        pooled_features = torch.transpose(pooled_features, 1, 2) # B, 25, 256

        # use exp_projected_feats and img_projected_feats to input to MLP or attention 
        # shape = 48, 256. B, 256        
        # output of this goes into the encoder

        img_exp_feature = self.early_attn(pooled_features, exp_feature) # B, 4, 256 (OUT)

        embed = self.vgtr(img_exp_feature, exp_feature, expression_word_id)
        embed2 = torch.cat([embed[:, i] for i in range(self.num_exp_tokens)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        return pred
