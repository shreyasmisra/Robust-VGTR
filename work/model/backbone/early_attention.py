import torch
import torch.nn as nn
import torch.nn.functional as F


class DotAttention(nn.Module):
    def __init__(self, dim = 256, l_norm=False):
        super(DotAttention, self).__init__()
        """
        attention = softmax(Q.K/sqrt(d)).V
        """
        self.l_norm = l_norm
        self.dim = dim
        if l_norm:
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, img_feats, exp_feats, mask=None):
        """
        img_features -> B, 25, 256. 
        exp_features -> B, 4, 256
        """
        
        q = exp_feats
        k = v = img_feats

        batch_size, queryL = q.size(0), q.size(1)
        batch_size, imgL = k.size(0), k.size(1)

        energy = torch.div(torch.bmm(k, torch.transpose(q, 1, 2)), torch.sqrt(self.dim)) # B, 25, 4

        if mask:
            energy.masked_fill_(mask, -1e9)

        energy = energy.view(batch_size*imgL, queryL)

        attn = nn.Softmax()(energy)
        attn = attn.view(batch_size, imgL, queryL) # B, 25, 4
        
        context = torch.bmm(torch.transpose(attn, 1, 2), v) # B, 4, 25 * B, 25, 256 -> B, 4, 256 (OUT)

        if self.l_norm:
            return self.norm(context)

        return context # B, 4, 256 (OUT)


class ImageTextStackedAttention(nn.Module):
    def __init__(self):
        super(ImageTextStackedAttention, self).__init__()
    
    def forward(self, img_feats, exp_feats):
        q = exp_feats
        k = v = img_feats

class TextImageStackedAttention(nn.Module):
    def __init__(self):
        super(TextImageStackedAttention, self).__init__()
    
    def forward(self, img_feats, exp_feats):
        q = exp_feats
        k = v = img_feats
