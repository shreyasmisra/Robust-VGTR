import numpy as np
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

        energy = torch.div(torch.bmm(k, torch.transpose(q, 1, 2)), np.sqrt(self.dim)) # B, 25, 4

        # add norm to energy ?
        if mask:
            energy.masked_fill_(mask, -1e9)

        energy = energy.view(batch_size*imgL, queryL)

        attn = F.softmax(energy, dim=1)
        attn = attn.view(batch_size, imgL, queryL) # B, 25, 4
        
        context = torch.bmm(torch.transpose(attn, 1, 2), v) # B, 4, 25 * B, 25, 256 -> B, 4, 256 (OUT)

        if self.l_norm:
            return self.norm(context)

        return context # B, 4, 256 (OUT)


class CosineAttention(nn.Module):
    def __init__(self, dim=256, l_norm=False):
        super(CosineAttention, self).__init__()
        self.l_norm = l_norm
        self.dim = dim
        if l_norm:
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, k, q):
        """
        img_features -> B, 25, 256. 
        exp_features -> B, 4, 256
        Both have to be the same shape for cosine similarity
        """
        
        v = k

        batch_size, queryL = q.size(0), q.size(1)
        batch_size, imgL = k.size(0), k.size(1)

        # this wont work as k and q should have matching shapes
        energy = F.cosine_similarity(k, torch.transpose(q, 1, 2)) # B, 25, 4

        energy = energy.view(batch_size*imgL, queryL)
        attention = F.softmax(energy, dim=1)
        
        attention = attention.view(batch_size, imgL, queryL) # B, 25, 4
        
        context = torch.bmm(torch.transpose(attention, 1, 2), v) # B, 4, 25 * B, 25, 256 -> B, 4, 256 (OUT)

        if self.l_norm:
            return self.norm(context)

        return context # B, 4, 256 (OUT)

class TextImageStackedAttention(nn.Module):
    def __init__(self):
        super(TextImageStackedAttention, self).__init__()
    
    def cosine_similarity(self, x1, x2, dim=1, eps=1e-8):
        w12 = torch.sum(x1 * x2, dim)
        w1 = torch.norm(x1, 2, dim)
        w2 = torch.norm(x2, 2, dim)
        return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

    def l2norm(self, X, dim, eps=1e-8):
        """L2-normalize columns of X
        """
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X
    
    def attention(self, q, context):
        
        batch_size, queryL = q.size(0), q.size(1)
        batch_size, imgL = context.size(0), context.size(1)

        qT = torch.transpose(q, 1, 2)
        attn = torch.bmm(context, qT)

        attn = nn.LeakyReLU(0.1)(attn)
        attn = self.l2norm(attn, 2)

        # --> (batch, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, imgL)
        attn = nn.Softmax()(attn)
        # --> (batch, queryL, sourceL)
        attn = attn.view(batch_size, queryL, imgL)
        # --> (batch, sourceL, queryL)
        attnT = torch.transpose(attn, 1, 2).contiguous()

        # --> (batch, d, sourceL)
        contextT = torch.transpose(context, 1, 2)
        # (batch x d x sourceL)(batch x sourceL x queryL)
        # --> (batch, d, queryL)
        weightedContext = torch.bmm(contextT, attnT)
        # --> (batch, queryL, d)
        weightedContext = torch.transpose(weightedContext, 1, 2)

        return weightedContext
    
    def forward(self, img_feats, exp_feats):
        q = exp_feats
        k = v = img_feats

        context = self.attention(exp_feats, img_feats)
        similarity = self.cosine_similarity(exp_feats, context, dim=2) # check the dims

class CoAttention(nn.Module):
    def __init__(self, dim=256):
        super(CoAttention, self).__init__()

        self.dim = dim
        self.attn = DotAttention(dim)
    
    def forward(self, img_feats, exp_feats):
        visual_attention  = self.attn.forward(img_feats, exp_feats)

        co_attn = self.attn.forward(exp_feats, visual_attention)

        return co_attn

