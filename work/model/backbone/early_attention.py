import torch
import torch.nn as nn
import torch.nn.functional as F


class DotAttention(nn.Module):
    def __init__(self, dim = 256, l_norm=False):
        super(DotAttention).__init__
        """
        attention = softmax(Q.K/sqrt(d)).V
        """
        self.l_norm = l_norm
        if l_norm:
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, img_feats, exp_feats, mask=None):
        """
        img_features -> (48, 256). 
        exp_features -> (48, 256). 
        """
        k = v = img_feats
        q = exp_feats

        energy = torch.div(torch.bmm(k, q), torch.sqrt(k.shape[1]))

        # optional: apply mask
        if mask:
            energy.masked_fill_(mask, -1e9)

        attn = F.softmax(energy, dim=1)

        context = torch.bmm(attn, v)

        if self.l_norm:
            return self.norm(context)

        return context


class ImageTextStackedAttention(nn.Module):
    def __init__(self):
        super(ImageTextStackedAttention).__init__
    
    def forward(self, img_feats, exp_feats):
        q = exp_feats
        k = v = img_feats

class TextImageStackedAttention(nn.Module):
    def __init__(self):
        super(TextImageStackedAttention).__init__
    
    def forward(self, img_feats, exp_feats):
        q = exp_feats
        k = v = img_feats
