# -*- coding: utf-8 -*-
from multiprocessing import pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.visual_backbone import build_visual_backbone
from .backbone.rnn import build_textual_encoder
from .vgtr.vgtr import build_vgtr

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
    
        self.conv = nn.Conv2d(2048, 256, kernel_size=1, padding=0)
        self.conv_text = nn.Conv1d(4, 256, kernel_size=1, padding=0)

    def forward(self, img, expression_word_id):

        img_feature = self.visual_encoder(img) #img_feature = (48, 2048, 16, 16)
        exp_feature = self.textual_encoder(expression_word_id) #exp_feature = (48, 4, 256)
        
        embed = self.vgtr(img_feature, exp_feature, None, expression_word_id)
        embed2 = torch.cat([embed[:, i] for i in range(self.num_exp_tokens)], dim=-1)

        pred = self.prediction_head(embed2).sigmoid()

        # for contrastive loss
        img_feature = self.conv(img_feature).flatten(2) 
        exp_feature = self.conv_text(exp_feature)

        return img_feature, exp_feature, pred
