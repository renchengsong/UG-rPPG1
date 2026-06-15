import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_
from neural_methods.layers.Mamba_EncDec import Encoder, EncoderLayer
from mamba_ssm import Mamba


# ================== FANLayer ==================
class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=True):
        super(FANLayer, self).__init__()
        p_dim = output_dim // 4
        ps_pair = p_dim * 2
        g_dim = output_dim - ps_pair
        assert g_dim > 0
        self.input_linear_p = nn.Linear(input_dim, p_dim, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, g_dim)
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)
        gate = torch.sigmoid(self.gate)
        output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        return output


# ================== Fusion_Stem ==================
class Fusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24, dropout=0.3):
        super(Fusion_Stem, self).__init__()
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D-2]], 1)
        x2 = torch.cat([x[:, :1], x[:, :D-1]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:], x[:, D-1:]], 1)
        x5 = torch.cat([x[:, 2:], x[:, D-1:], x[:, D-1:]], 1)

        x_diff = self.stem12(torch.cat([x2-x1, x3-x2, x4-x3, x5-x4], 2).view(N*D, 12, H, W))
        x3 = x3.contiguous().view(N*D, C, H, W)
        x = self.stem11(x3)
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2
        return x


# ================== Attention Mask ==================
class Attention_mask(nn.Module):
    def forward(self, x):
        xsum = x.sum(dim=(3, 4), keepdim=True)
        return x / xsum * x.shape[3] * x.shape[4] * 0.5


# ================== Mamba Layer ==================
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, dropout=0.1, activation='gelu', e_layers=2, expand=2, d_conv=4):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.encoder = Encoder([
            EncoderLayer(
                Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand),
                Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand),
                dim,
                int(dim * 2),
                dropout=dropout,
                activation=activation
            ) for _ in range(e_layers)
        ], norm_layer=nn.LayerNorm(dim))

    def forward(self, x):
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = self.norm((x - means) / stdev)
        enc_out, _ = self.encoder(x, attn_mask=None)
        return enc_out * stdev + means


# ================== Block Mamba ==================
class Block_mamba(nn.Module):
    def __init__(self, dim, mlp_ratio=2, drop_path=0., norm_layer=nn.LayerNorm, dropout=0.3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim, dropout=dropout)
        self.mlp = FANLayer(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x + self.drop_path(self.dropout(self.attn(self.norm1(x))))
        x = x + self.drop_path(self.dropout(self.mlp(self.norm2(x))))
        return x


# ================== SimpleMamba_MC ==================
class SimpleMamba(nn.Module):
    def __init__(self,
                 depth=24,
                 embed_dim=96,
                 mlp_ratio=2,
                 drop_rate=0.3,
                 drop_path_rate=0.3,
                 use_mc_dropout=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_mc_dropout = use_mc_dropout

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim//4, dropout=drop_rate)
        self.attn_mask = Attention_mask()
        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim//4, embed_dim, kernel_size=(2,5,5),
                      stride=(2,1,1), padding=(0,2,2)),
            nn.BatchNorm3d(embed_dim),
            nn.Dropout3d(drop_rate)
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block_mamba(embed_dim, mlp_ratio, drop_path=dpr[i], dropout=drop_rate)
            for i in range(depth)
        ])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Sequential(
            nn.Conv1d(embed_dim, 1, kernel_size=1),
            nn.Dropout(drop_rate)
        )

    def forward_once(self, x):
        B, D, C, H, W = x.shape
        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim//4, H//8, W//8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)
        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask
        x = torch.mean(x, dim=[3, 4])
        x = rearrange(x, 'b c t -> b t c')
        for blk in self.blocks:
            x = blk(x)
        rPPG = x.permute(0, 2, 1)
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG).squeeze(1)
        return rPPG

    # 🧠 Monte Carlo inference
    def forward_mc(self, x, n_samples=20):
        preds = []
        training_state = self.training
        self.train(True)  # ensure dropout active even in eval mode
        for _ in range(n_samples):
            preds.append(self.forward_once(x).unsqueeze(0))
        preds = torch.cat(preds, dim=0)
        mean = preds.mean(0)
        var = preds.var(0)
        self.train(training_state)  # restore state
        return mean, var

    def forward(self, x):
        return self.forward_once(x)
