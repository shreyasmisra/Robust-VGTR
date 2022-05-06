import torch.nn as nn
from .vg_encoder import VGEncoder
from .vg_decoder import VGDecoder
from .position_encoding import PositionEmbeddingSine, PositionEncoding1D
#from .vg_encoder_without_cross_fusion import VGEncoder


class VGTR(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.input_proj = nn.Conv2d(2048, args.hidden_dim, kernel_size=1)

        self.encoder = VGEncoder(
            d_model=args.hidden_dim,
            dropout=args.dropout,
            nhead=args.nheads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.enc_layers)

        self.decoder = VGDecoder(n_layers=args.dec_layers,
                                 n_heads=args.nheads,
                                 d_model=args.hidden_dim)

    def forward(self, img_exp_fused, sent, pos_feature, sent_id):
        """
        img_exp_fused -> B, 4, 256
        sent -> B, 4, 256
        """

        # encoder
        # fused_vis_feature, fused_exp_feature = self.encoder(self.input_proj(img_feature), pos_feature, sent)
        fused_vis_feature, fused_exp_feature = self.encoder(img_exp_fused, pos_feature, sent)
        
        # decoder
        out = self.decoder(fused_vis_feature.transpose(0, 1), fused_exp_feature,
                           pos_feature=None)
                           
        return out.transpose(0, 1)


def build_vgtr(args):

    return VGTR(args)