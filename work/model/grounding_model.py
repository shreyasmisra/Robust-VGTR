# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from .backbone.visual_backbone import build_visual_backbone
from .backbone.rnn import build_textual_encoder
from .vgtr.vgtr import build_vgtr


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

        self.conv = nn.Conv2d(2048, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, img, expression_word_id):

        img_feature, pooled_features = self.visual_encoder(img) #img_feature = (48, 2048, 16, 16)
        exp_feature = self.textual_encoder(expression_word_id) #exp_feature = (48, 4, 256)
        
        img_projected_feats = [] # each element is 48, 256
        for i in range(len(pooled_features)):
            pooled_feats_flatten = pooled_features[i].flatten(2).flatten(1)
            img_projected_feats.append(self.pooled_feats_linear(pooled_feats_flatten))

        exp_projected_feats = self.exp_feats_linear(exp_feature.flatten(1)) # 48, 256

        # use exp_projected_feats and img_projected_feats to input to MLP or attention 
        # shape = 48, 256. B, 256        
        # output of this goes into the encoder

        embed = self.vgtr(img_feature, exp_feature, expression_word_id)
        embed2 = torch.cat([embed[:, i] for i in range(self.num_exp_tokens)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        img_feature = self.pool(self.conv(img_feature)).flatten(2) # 48, 256, 4
        exp_feature = exp_feature#.transpose(1, 2) # 48, 4, 256

        # F.cosine_similarity(dim=2) # 48, 256

        return img_feature, exp_feature, pred
