# -*- coding: utf-8 -*-
from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.visual_backbone import build_visual_backbone
from .backbone.rnn import build_textual_encoder
from .vgtr.vgtr import build_vgtr

from .backbone.early_attention import DotAttention, CosineAttention, CoAttention
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

        #self.pooled_feats_linear = nn.Linear(2048, 256) # make part of args
        #self.exp_feats_linear = nn.Linear(1024, 256)

        self.pos_encoder = PositionEmbeddingSine(args.hidden_dim//2, normalize=False)
        self.pos_encoder_1d = PositionEncoding1D(256)
        self.pointwise = nn.Conv2d(128, 256, kernel_size=1, stride=1)

        self.early_attn = DotAttention(l_norm=False)
        self.cosine_attn = CosineAttention()
        self.co_attn = CoAttention()

        self.conv = nn.Conv2d(2048, 256, kernel_size=1, padding=0)
        self.conv_text = nn.Conv1d(4, 256, kernel_size=1, padding=0)

    def forward(self, img, expression_word_id):

        img_feature = self.visual_encoder(img) #img_feature = (48, 2048, 16, 16)
        exp_feature = self.textual_encoder(expression_word_id) #exp_feature = (48, 4, 256)
        
        # img_exp_feature = []

        # # increasing number of channels
        # x_high_channels = []
        # for feat in pooled_features:
        #     x_high_channels.append(self.pointwise(feat)) # B, 256, 4, 4

        # # finding positional encoding
        # pos_embed = []
        # for feat in x_high_channels:
        #     pos_encode = self.pos_encoder(feat)
        #     #print('pos embed out', pos_encode.shape)
        #     pos_embed.append(pos_encode)

        # # calculating cross-attention
        # for i, feat in enumerate(x_high_channels):           
        
        #     feat = feat + pos_embed[i] # early addition.  
        #     x = feat.flatten(2) # B, 256, 16

        #     x = torch.transpose(x, 1, 2) # B, 16, 256
        #     img_exp_feature.append(self.co_attn(x, exp_feature)) # [(B, 4, 256), ... ]

        # img_exp_feature = torch.cat(img_exp_feature, dim=1) # B, 16, 256
        #img_exp_feature = torch.sigmoid(img_exp_feature)
        #pos_embed = torch.stack(pos_embed, dim=3)
        
        # embed = self.vgtr(img_feature, img_exp_feature, exp_feature, None, expression_word_id)
        embed = self.vgtr(img_feature, exp_feature, None, expression_word_id)
        # print(embed.shape)
        embed2 = torch.cat([embed[:, i] for i in range(self.num_exp_tokens)], dim=-1)
        # print(embed2.shape)

        pred = self.prediction_head(embed2).sigmoid()

        # for contrastive loss
        img_feature = self.conv(img_feature).flatten(2) 
        exp_feature = self.conv_text(exp_feature)

        return img_feature, exp_feature, pred
