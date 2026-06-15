"Uncertainty-Guided Remote Photoplethysmography for Robust Pulse Estimation"
import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
from functools import partial
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
import math
from einops import rearrange
# from mamba_ssm.modules.mamba_simple import Mamba
import torch
import torch.nn as nn
from neural_methods.layers.Mamba_EncDec import Encoder, EncoderLayer
from mamba_ssm import Mamba


class FANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, with_gate=True):
        super(FANLayer, self).__init__()
        self.input_linear_p = nn.Linear(input_dim, output_dim // 4, bias=bias)
        self.input_linear_g = nn.Linear(input_dim, (output_dim - output_dim // 2))
        self.activation = nn.GELU()
        if with_gate:
            self.gate = nn.Parameter(torch.randn(1, dtype=torch.float32))

    def forward(self, src):
        g = self.activation(self.input_linear_g(src))
        p = self.input_linear_p(src)

        if not hasattr(self, 'gate'):
            output = torch.cat((torch.cos(p), torch.sin(p), g), dim=-1)
        else:
            gate = torch.sigmoid(self.gate)
            output = torch.cat((gate * torch.cos(p), gate * torch.sin(p), (1 - gate) * g), dim=-1)
        return output


class Fusion_Stem(nn.Module):
    def __init__(self, apha=0.5, belta=0.5, dim=24):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(dim // 2, eps=1e-05, momentum=0.1, affine=True,
                                                   track_running_stats=True),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                                    )

        self.stem12 = nn.Sequential(nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3),
                                    nn.BatchNorm2d(dim // 2),
                                    nn.ReLU(inplace=True),
                                    nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
                                    )

        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/8,W/8]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:, :1, :, :, :], x[:, :1, :, :, :], x[:, :D - 2, :, :, :]], 1)
        x2 = torch.cat([x[:, :1, :, :, :], x[:, :D - 1, :, :, :]], 1)
        x3 = x
        x4 = torch.cat([x[:, 1:, :, :, :], x[:, D - 1:, :, :, :]], 1)
        x5 = torch.cat([x[:, 2:, :, :, :], x[:, D - 1:, :, :, :], x[:, D - 1:, :, :, :]], 1)
        x_diff = self.stem12(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        # fusion layer1
        x_path1 = self.apha * x + self.belta * x_diff
        x_path1 = self.stem21(x_path1)
        # fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha * x_path1 + self.belta * x_path2

        return x


class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, dropout=0.1, activation='gelu', e_layers=2, expand=2, d_conv=4):
        super().__init__()
        self.dim = dim
        self.use_norm = True  # 默认使用归一化
        self.norm = nn.LayerNorm(dim)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    Mamba(
                        d_model=dim,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    dim,
                    int(dim * 2),  # 假设 mlp_ratio 为 2，可根据需要调整
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(dim)
        )

    def forward(self, x):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        x = self.norm(x)
        enc_out, attns = self.encoder(x, attn_mask=None)

        if self.use_norm:
            # 确保 enc_out 和 stdev 的形状一致
            enc_out = enc_out * (stdev.expand(-1, enc_out.shape[1], -1))
            enc_out = enc_out + (means.expand(-1, enc_out.shape[1], -1))

        return enc_out

class Block_mamba(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 d_state=48,
                 dropout=0.1,
                 activation='gelu',
                 e_layers=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim, d_state, dropout, activation, e_layers)
        # 替换 Frequencydomain_FFN 为 FANLayer
        self.mlp = FANLayer(input_dim=dim, output_dim=dim, bias=True, with_gate=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if isinstance(self.drop_path, nn.Identity):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

# class DenseNormalGamma(nn.Module):
#     def __init__(self, units):
#         super(DenseNormalGamma, self).__init__()
#         self.units = int(units)
#         self.dense = nn.Linear(self.units, 4 * self.units, bias=True)
#
#     def evidence(self, x):
#         return F.softplus(x)
#
#     def forward(self, x):
#         output = self.dense(x)
#         mu, logv, logalpha, logbeta = torch.split(output, self.units, dim=-1)
#         v = self.evidence(logv)
#         alpha = self.evidence(logalpha) + 1
#         beta = self.evidence(logbeta)
#         return torch.cat([mu, v, alpha, beta], dim=-1)


# class DenseNormalGamma(nn.Module):
#     def __init__(self, units, eps=1e-6, alpha_offset=1.0):
#         """
#         参数:
#             units: 输入特征维度
#             eps: 数值稳定性截断阈值
#             alpha_offset: alpha参数偏移量（默认1.0确保alpha>1）
#         """
#         super().__init__()
#         self.units = int(units)
#         self.eps = eps
#         self.alpha_offset = alpha_offset
#
#         # 使用 Xavier 初始化并添加偏置!!!!!!!
#         self.dense = nn.Linear(self.units, 4 * self.units, bias=True)
#         nn.init.xavier_uniform_(self.dense.weight)
#         nn.init.zeros_(self.dense.bias)
#
#     def evidence(self, x):
#         """带截断的 Softplus 激活"""
#         return F.softplus(x).clamp(min=self.eps, max=1e6)
#
#     def forward(self, x):
#         """
#         输入: [batch_size, units]
#         输出: [batch_size, 4*units] (mu, v, alpha, beta)
#         """
#         # 线性映射
#         output = self.dense(x)  # [batch_size, 4*units]
#
#         # 拆分参数
#         mu, logv, logalpha, logbeta = torch.split(output, self.units, dim=-1)
#
#         # 参数转换与稳定性处理
#         v = self.evidence(logv)  # 精度 (正数)
#         alpha = self.evidence(logalpha) + self.alpha_offset  # 形状参数 (>alpha_offset)
#         beta = self.evidence(logbeta)  # 尺度参数 (正数)
#
#         # 拼接输出
#         return torch.cat([mu, v, alpha, beta], dim=-1)

# class NormalInvGamma(nn.Module):
#     def __init__(self, in_features, out_units):
#         super().__init__()
#         self.dense = nn.Linear(in_features, out_units * 4)
#         self.out_units = out_units
#
#     def evidence(self, x):
#         return F.softplus(x)
#
#     def forward(self, x):
#         out = self.dense(x)
#         mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
#         v = self.evidence(logv)
#         alpha = self.evidence(logalpha) + 1
#         beta = self.evidence(logbeta)
#         return mu, v, alpha, beta

#
class NormalInvGamma(nn.Module):
    def __init__(self, in_features, out_units):
        super().__init__()
        self.dense = nn.Linear(in_features, out_units * 4)
        self.out_units = out_units
        # 使初始化与Keras Dense一致#################3
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        out = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return mu, v, alpha, beta
#

class LogNormalInvGamma(nn.Module):
    """
    输出 μ, logv, logα, logβ
    再映射到 v, α, β 的正数空间
    """
    def __init__(self, in_features, out_units,
                 eps=1e-6, alpha_offset=1.0):
        super().__init__()
        self.dense = nn.Linear(in_features, 4 * out_units)
        self.out_units = out_units
        self.eps = eps
        self.alpha_offset = alpha_offset

        # 与原来一致的初始化
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x):
        out = self.dense(x)                    # [B, T, 4*out_units]
        mu, logv, logalpha, logbeta = torch.chunk(out, 4, dim=-1)

        # 稳定映射到正数空间
        v     = F.softplus(logv)     + self.eps
        alpha = F.softplus(logalpha) + self.alpha_offset
        beta  = F.softplus(logbeta)  + self.eps

        return mu, v, alpha, beta

# class NormalInvGamma(nn.Module):
#     def __init__(self, in_features, out_units):
#         super().__init__()
#         self.dense = nn.Linear(in_features, out_units * 4)
#         self.out_units = out_units
#         self.eps = 1e-2  # 更大的下限
#         self.alpha_offset = 1.1  # 保证alpha>1.1
#         self.beta_offset = 1e-2  # 保证beta>1e-2
#         self.v_offset = 1e-2     # 保证v>1e-2
#
#         # 使初始化与Keras Dense一致
#         nn.init.xavier_uniform_(self.dense.weight)
#         nn.init.zeros_(self.dense.bias)
#
#     def evidence(self, x):
#         return F.softplus(x) + self.eps  # 保证输出不会太小
#
#     def forward(self, x):
#         out = self.dense(x)
#         mu, logv, logalpha, logbeta = torch.split(out, self.out_units, dim=-1)
#         v = self.evidence(logv) + self.v_offset
#         alpha = self.evidence(logalpha) + self.alpha_offset
#         beta = self.evidence(logbeta) + self.beta_offset
#         return mu, v, alpha, beta


class SimpleMamba(nn.Module):
    def __init__(self,
                 depth=24,
                 embed_dim=96,
                 mlp_ratio=2,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 initializer_cfg=None,
                 device=None,
                 dtype=None,
                 output_units= 1,  # 输出维度（例如心率预测为1）
                 **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.embed_dim = embed_dim

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)
        self.attn_mask = Attention_mask()

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim, kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([Block_mamba(
            dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path=inter_dpr[i],
            norm_layer=nn.LayerNorm, )
            for i in range(depth)])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, output_units, kernel_size=1, stride=1, padding=0)

        # 添加 DenseNormalGamma 层
        # self.uncertainty_layer = DenseNormalGamma(units=output_units)
        # self.uncertainty_layer = DenseNormalGamma(units=output_units, eps=1e-4, alpha_offset=0.1)
        # self.uncertainty_layer = NormalInvGamma(in_features=1, out_units=output_units)
        self.uncertainty_layer = LogNormalInvGamma(in_features=1, out_units=output_units)

        # init
        self.apply(segm_init_weights)
        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

        # 2. evidential 头单独初始化
        with torch.no_grad():
            nn.init.xavier_uniform_(self.uncertainty_layer.dense.weight)
            nn.init.zeros_(self.uncertainty_layer.dense.bias)

        # with torch.no_grad():
        #     self.uncertainty_layer.dense.reset_parameters()

        # nn.init.kaiming_uniform_(self.uncertainty_layer.dense.weight, nonlinearity='relu')
        # nn.init.zeros_(self.uncertainty_layer.dense.bias)

    def forward(self, x):
        B, D, C, H, W = x.shape

        x = self.Fusion_Stem(x)  # [N*D C H/8 W/8]
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, 4)
        x = torch.mean(x, 3)
        x = rearrange(x, 'b c t -> b t c')

        for blk in self.blocks:
            x = blk(x)

        rPPG = x.permute(0, 2, 1)   # [N, D, C]-> [N,C,D]
        rPPG = self.upsample(rPPG)  # [N, C, T]
        rPPG = self.ConvBlockLast(rPPG)  # [N, output_units, t],output_units=1
        rPPG = rPPG.permute(0, 2, 1)


        mu, v, alpha, beta = self.uncertainty_layer(rPPG)
        uncertainty_params = torch.cat([mu, v, alpha, beta], dim=-1)

        return uncertainty_params


    # def forward(self, x):
    #     B, D, C, H, W = x.shape
    #
    #     x = self.Fusion_Stem(x)  # [N*D C H/8 W/8]
    #     x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
    #     x = self.stem3(x)
    #
    #     mask = torch.sigmoid(x)
    #     mask = self.attn_mask(mask)
    #     x = x * mask
    #
    #     x = torch.mean(x, 4)
    #     x = torch.mean(x, 3)
    #     x = rearrange(x, 'b c t -> b t c')
    #
    #     for blk in self.blocks:
    #         x = blk(x)
    #
    #     rPPG = x.permute(0, 2, 1)   # [N, D, C]-> [N,C,D]
    #     rPPG = self.upsample(rPPG)  # [N, C, T]
    #     rPPG = self.ConvBlockLast(rPPG)  # [N, output_units, t],output_units=1
    #     # print("rPPG 输出形状:", rPPG.shape)
    #     rPPG = rPPG.permute(0, 2, 1)
    #     # print("rPPG 输出形状:", rPPG.shape)
    #
    #     mu, v, alpha, beta = self.uncertainty_layer(rPPG)
    #     uncertainty_params = torch.cat([mu, v, alpha, beta], dim=-1)
    #     # print("mu输出形状:", mu.shape)
    #     # print("uncertainty_params输出形状:", uncertainty_params.shape)
    #
    #     # rPPG = self.ConvBlockLast(rPPG)  # [N, output_units, t],output_units=1
    #     # uncertainty_params = rPPG.permute(0, 2, 1)
    #     # print("uncertainty_params输出形状:", uncertainty_params.shape)
    #     # print("rPPG 输出形状:", rPPG .shape)
    #
    #     # 通过 DenseNormalGamma 层输出不确定性参数
    #     # uncertainty_params = self.uncertainty_layer(rPPG.permute(0, 2, 1))  # [N, D, 4*output_units]
    #     # print("uncertainty_params输出形状:", uncertainty_params.shape)
    #     # return rPPG.squeeze(1), uncertainty_params  # 返回预测值和不确定性参数
    #     return uncertainty_params
