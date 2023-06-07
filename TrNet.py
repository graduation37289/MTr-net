import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

__all__ = ['TrNet']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule
import pdb


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, M):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, M)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=6, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size, 4)
        patch_size = (patch_size, patch_size, 1)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W, self.M = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[
            2]
        self.num_patches = self.H * self.W * self.M
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, 0))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W, M = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W, M


# class NoBottleneck(nn.Module):
#     def __init__(self, input, output):
#         super(NoBottleneck, self).__init__()
#         self.relu = nn.ReLU(inplace=True)

#         self.gn1 = nn.BatchNorm2d(output)
#         self.conv1 = nn.Conv2d(input, output, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

#         self.gn2 = nn.BatchNorm2d(output)
#         self.conv2 = nn.Conv2d(output, output, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)

#     def forward(self, x):
#         skip = x
#         seg = self.conv1(x)
#         seg = self.gn1(seg)
#         seg = self.relu(seg)

#         seg = self.conv2(seg)
#         seg = self.gn2(seg)
#         seg = self.relu(seg)

#         seg = seg + skip
#         return seg

class NoBottleneck(nn.Module):
    def __init__(self, input, output):
        super(NoBottleneck, self).__init__()
        self.gelu = nn.GELU()

        self.gn1 = nn.BatchNorm3d(output)
        self.conv1 = nn.Conv3d(input, output, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), dilation=1,
                               bias=False)

        self.conv1_1 = nn.Conv3d(output, output, kernel_size=(1, 1, 1), stride=1, padding=0, bias=False)
        self.gn2 = nn.BatchNorm3d(output)

    def forward(self, x):
        # print('x', x.shape)
        skip = x
        seg = self.conv1(x)
        seg = self.gn1(seg)
        # print('seg3*3', seg.shape)
        seg = self.conv1_1(seg)
        seg = self.gelu(seg)
        # print('seg1*1', seg.shape)
        seg = self.conv1_1(seg)
        seg = self.gn2(seg)
        # print('seg', seg.shape)
        seg = seg + skip
        seg = self.gelu(seg)
        # print('seg', seg.shape)
        return seg


class Attention_Gate(nn.Module):
    def __init__(self, input, output):
        super(Attention_Gate, self).__init__()
        C = input // 2
        self.gate = nn.Linear(C, 3 * C)
        self.linear1 = nn.Linear(C, C)
        self.linear2 = nn.Linear(C, C)
        self.conv1x1 = nn.Conv3d(input, output, kernel_size=1)

    def forward(self, x1, x2):
        #         print('x1, x2', x1.shape, x2.shape)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        diff_z = x2.size()[4] - x1.size()[4]
        # print('diff_x, diff_y', diff_x, diff_y)

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2,
                        diff_z // 2, diff_z - diff_z // 2])
        # print('x1', x1.shape)
        # attention
        B, C, H, W, M = x1.shape
        x1 = x1.permute(0, 2, 3, 4, 1)
        x2 = x2.permute(0, 2, 3, 4, 1)
        # print('gate', self.gate(x1).shape)
        gate = self.gate(x1).reshape(B, H, W, M, 3, C).permute(4, 0, 1, 2, 3, 5)
        g1, g2, g3 = gate[0], gate[1], gate[2]
        x2 = torch.sigmoid(self.linear1(g1 + x2)) * x2 + torch.sigmoid(g2) * torch.tanh(g3)
        x2 = self.linear2(x2)
        x1 = x1.permute(0, 4, 1, 2, 3)
        x2 = x2.permute(0, 4, 1, 2, 3)

        x = self.conv1x1(torch.cat([x2, x1], dim=1))
        return x


class EfficientAttention(nn.Module):

    def __init__(self, in_channels, key_channels, value_channels, head_count=1):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = nn.Conv3d(in_channels, key_channels, 1)
        self.queries = nn.Conv3d(in_channels, key_channels, 1)
        self.values = nn.Conv3d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv3d(value_channels, in_channels, 1)

    def forward(self, input_):
        n, _, h, w, m = input_.size()

        keys = self.keys(input_).reshape((n, self.key_channels, h * w * m))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w * m)
        values = self.values(input_).reshape((n, self.value_channels, h * w * m))

        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)

            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)

            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_value_channels, h, w, m)  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        attention = self.reprojection(aggregated_values)

        return attention


class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W, M) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W, M) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class ChannelAttention(nn.Module):
    """
    Input -> x: [B, N, C]
    Output -> [B, N, C]
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """x: [B, N, C]"""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        # -------------------
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        # ------------------
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DualTransformerBlock(nn.Module):
    """
    Input  -> x (Size: (b, (H*W), d)), H, W
    Output -> (b, (H*W), d)
    """

    def __init__(self, in_dim, key_dim, value_dim, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = EfficientAttention(in_channels=in_dim, key_channels=key_dim, value_channels=value_dim, head_count=1)
        self.norm2 = nn.LayerNorm(in_dim)
        self.norm3 = nn.LayerNorm(in_dim)
        self.channel_attn = ChannelAttention(in_dim)
        self.norm4 = nn.LayerNorm(in_dim)
        if token_mlp == "mix":
            self.mlp1 = MixFFN(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN(in_dim, int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp1 = MixFFN_skip(in_dim, int(in_dim * 4))
            self.mlp2 = MixFFN_skip(in_dim, int(in_dim * 4))
        else:
            self.mlp1 = MLP_FFN(in_dim, int(in_dim * 4))
            self.mlp2 = MLP_FFN(in_dim, int(in_dim * 4))

    def forward(self, x: torch.Tensor, H, W, M) -> torch.Tensor:
        # dual attention structure, efficient attention first then transpose attention
        norm1 = self.norm1(x)
        # print('norm1', norm1.shape)
        b, hwm, d = norm1.shape
        norm1 = norm1.reshape(b, H, W, M, d).permute(0, 4, 1, 2, 3)

        attn = self.attn(norm1)
        b, d, h, w, m = attn.shape
        attn = attn.reshape(b, h*w*m, d)

        add1 = x + attn
        norm2 = self.norm2(add1)
        mlp1 = self.mlp1(norm2, H, W, M)

        add2 = add1 + mlp1
        norm3 = self.norm3(add2)
        channel_attn = self.channel_attn(norm3)

        add3 = add2 + channel_attn
        norm4 = self.norm4(add3)
        mlp2 = self.mlp2(norm4, H, W, M)

        mx = add3 + mlp2
        return mx


class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, height, width, m, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.height = height
        self.width = width
        self.m = m

        self.reprojection = nn.Conv3d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

    # x2 should be higher-level representation than x1
    def forward(self, x1, x2):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)

        # Re-arrange into a (Batch, Embedding dim, Tokens)
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]
            context = key @ value.transpose(1, 2)  # dk*dv
            attended_value = context.transpose(1, 2) @ query  # n*dv
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, self.height, self.width, self.m)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        reprojected_value = self.norm(reprojected_value)

        return reprojected_value


class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """

    def __init__(self, in_dim, key_dim, value_dim, height, width, m, head_count=1, token_mlp="mix"):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.H = height
        self.W = width
        self.M = m
        self.attn = Cross_Attention(key_dim, value_dim, height, width, m, head_count=head_count)
        self.norm2 = nn.LayerNorm((in_dim * 2))
        if token_mlp == "mix":
            self.mlp = MixFFN((in_dim * 2), int(in_dim * 4))
        elif token_mlp == "mix_skip":
            self.mlp = MixFFN_skip((in_dim * 2), int(in_dim * 4))
        else:
            self.mlp = MLP_FFN((in_dim * 2), int(in_dim * 4))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        # print('norm_1, norm_2', norm_1.shape, norm_2.shape)
        attn = self.attn(norm_1, norm_2)
        # attn = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(attn)

        # residual1 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x1)
        # residual2 = Rearrange('b (h w) d -> b h w d', h=self.H, w=self.W)(x2)
        residual = torch.cat([x1, x2], dim=2)
        tx = residual + attn
        mx = tx + self.mlp(self.norm2(tx), self.H, self.W, self.M)
        return mx


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 4 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W, M = self.input_resolution
        x = self.expand(x)

        B, L, C = x.shape
        # print(x.shape)
        assert L == H * W * M, "input feature has wrong size"

        x = x.view(B, H, W, M, C)
        # print('x', x.shape)
        x = x.reshape(B, H*2, W*2, M*2, C//8)
        x = x.view(B, -1, C // 8)
        x = self.norm(x.clone())

        return x


class MyDecoderLayer(nn.Module):
    def __init__(
        self, input_size, in_out_chan, head_count, token_mlp_mode, n_class=9, norm_layer=nn.LayerNorm, is_last=False
    ):
        super().__init__()
        dims = in_out_chan[0]
        out_dim = in_out_chan[1]
        key_dim = in_out_chan[2]
        value_dim = in_out_chan[3]
        x1_dim = in_out_chan[4]
        if not is_last:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims, key_dim, value_dim, input_size[0], input_size[1], input_size[2], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(2 * dims, out_dim)
            # transformer decoder
            self.layer_up = PatchExpand(input_resolution=input_size, dim=out_dim, dim_scale=2, norm_layer=norm_layer)
            self.last_layer = None
        else:
            self.x1_linear = nn.Linear(x1_dim, out_dim)
            self.cross_attn = CrossAttentionBlock(
                dims * 2, key_dim, value_dim, input_size[0], input_size[1], input_size[2], head_count, token_mlp_mode
            )
            self.concat_linear = nn.Linear(4 * dims, out_dim)
            # transformer decoder
            self.layer_up = FinalPatchExpand_X4(
                input_resolution=input_size, dim=out_dim, dim_scale=4, norm_layer=norm_layer
            )
            self.last_layer = nn.Conv2d(out_dim, n_class, 1)

        self.layer_former_1 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)
        self.layer_former_2 = DualTransformerBlock(out_dim, key_dim, value_dim, head_count, token_mlp_mode)

        def init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        init_weights(self)

    def forward(self, x1, x2=None):
        # print('x1', x1.shape)
        if x2 is not None:  # skip connection exist
            b, h, w, m, c = x2.shape
            x2 = x2.view(b, -1, c)
            x1_expand = self.x1_linear(x1)
            cat_linear_x = self.concat_linear(self.cross_attn(x1_expand, x2))
            tran_layer_1 = self.layer_former_1(cat_linear_x, h, w, m)
            tran_layer_2 = self.layer_former_2(tran_layer_1, h, w, m)

            if self.last_layer:
                out = self.last_layer(self.layer_up(tran_layer_2).view(b, 4 * h, 4 * w, -1).permute(0, 3, 1, 2))
            else:
                out = self.layer_up(tran_layer_2)
        else:
            out = self.layer_up(x1)
        return out


class TrNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()
        blocks = [1, 2, 2]
        blocksup = [2, 2, 2, 2, 1]
        self.encoder1 = nn.Conv3d(1, 32, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.encoder2 = nn.Conv3d(32, 64, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.encoder3 = nn.Conv3d(64, embed_dims[0], (3, 3, 3), stride=1, padding=(1, 1, 1))

        self.ebn1 = nn.BatchNorm3d(32)
        self.ebn2 = nn.BatchNorm3d(64)
        self.ebn3 = nn.BatchNorm3d(embed_dims[0])

        self.layer0 = self._make_layer(NoBottleneck, 32, 32, blocks[0])
        self.layer1 = self._make_layer(NoBottleneck, 64, 64, blocks[1])
        self.layer2 = self._make_layer(NoBottleneck, embed_dims[0], embed_dims[0], blocks[2])
        
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        layers = [2, 2, 2]
        self.block1 = nn.ModuleList([DualTransformerBlock(embed_dims[1], embed_dims[1], embed_dims[1], head_count=1, token_mlp="mix_skip") for _ in range(layers[1])])
        
        self.block2 = nn.ModuleList([DualTransformerBlock(embed_dims[2], embed_dims[2], embed_dims[2], head_count=1, token_mlp="mix_skip") for _ in range(layers[2])])

        d_base_feat_size = 7  # 16 for 512 input size, and 7 for 224
        in_out_chan = [
            [64, 128, 128, 128, 160],
            [320, 320, 320, 320, 256],
            [512, 512, 512, 512, 512],
        ]  # [dim, out_dim, key_dim, value_dim, x2_dim]
        self.decoder_Trans1 = MyDecoderLayer(
            (d_base_feat_size * 1, d_base_feat_size * 1, 1),
            in_out_chan[2],
            head_count=1,
            token_mlp_mode="mix_skip",
            n_class=num_classes+1,
        )

        self.decoder_Trans2 = MyDecoderLayer(
            (d_base_feat_size * 2, d_base_feat_size * 2, 2),
            in_out_chan[1],
            head_count=1,
            token_mlp_mode="mix_skip",
            n_class=num_classes+1,
        )

        self.dblock1 = nn.ModuleList([DualTransformerBlock(embed_dims[1], embed_dims[1], embed_dims[1], head_count=1, token_mlp="mix_skip") for _ in range(layers[2])])

        self.dblock2 = nn.ModuleList([DualTransformerBlock(embed_dims[0], embed_dims[0], embed_dims[0], head_count=1, token_mlp="mix_skip") for _ in range(layers[2])])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        
        self.decoder1 = nn.Conv3d(embed_dims[2], embed_dims[1], (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.decoder2 = nn.Conv3d(embed_dims[1], embed_dims[0], (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.decoder3 = nn.Conv3d(embed_dims[0], 64, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.decoder4 = nn.Conv3d(64, 32, (3, 3, 3), stride=1, padding=(1, 1, 1))
        self.decoder5 = nn.Conv3d(32, 32, (3, 3, 3), stride=1, padding=(1, 1, 1))

        self.dbn1 = nn.BatchNorm3d(embed_dims[1])
        self.dbn2 = nn.BatchNorm3d(embed_dims[0])
        self.dbn3 = nn.BatchNorm3d(64)
        self.dbn4 = nn.BatchNorm3d(32)
        
        self.layerup1 = self._make_layer(NoBottleneck, embed_dims[1], embed_dims[1], blocksup[0])
        self.layerup2 = self._make_layer(NoBottleneck, embed_dims[0], embed_dims[0], blocksup[1])
        self.layerup3 = self._make_layer(NoBottleneck, 64, 64, blocksup[2])
        self.layerup4 = self._make_layer(NoBottleneck, 32, 32, blocksup[3])

        self.final = nn.Conv3d(32, num_classes, kernel_size=(1, 1, 1))

        self.soft = nn.Softmax(dim=1)

    def _make_layer(self, block, inplanes, outplanes, blocks):
        layers = []
        layers.append(block(inplanes, outplanes))
        for i in range(1, blocks):
            layers.append(
                block(inplanes, outplanes))
        return nn.Sequential(*layers)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), (2, 2, 1), (2, 2, 1)))
        # print('out', out.shape)
        out = self.layer0(out)
        # print('out', out.shape)
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), (2, 2, 1), (2, 2, 1)))
        out = self.layer1(out)
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), (2, 2, 1), (2, 2, 1)))
        out = self.layer2(out)
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W, M = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W, M)
        out = self.norm3(out)
        out = out.reshape(B, H, W, M, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out
        # print('t4', t4.shape)
        ### Bottleneck

        out, H, W, M = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W, M)
        out = self.norm4(out)
        out = out.reshape(B, H, W, M, -1).permute(0, 4, 1, 2, 3).contiguous()
        # print('out', out.shape)

        ### Stage 4
        # b, c, _, _, _ = out.shape
        # out = self.decoder_Trans1(out.permute(0, 2, 3, 4, 1).view(b, -1, c))
        # out = self.decoder_Trans2(out, t4.permute(0, 2, 3, 4, 1))
        # b, _, c = out.shape
        # out = out.permute(0, 2, 1).reshape(b, c, H*4, W*4, M*4)
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = self.attention_gate1(out, t4)
        out = self.layerup1(out)
        out = torch.add(out, t4)
        _, _, H, W, M = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W, M)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, M, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2, 2), mode='trilinear'))
        #         out = self.attention_gate2(out, t3)
        out = self.layerup2(out)
        out = torch.add(out, t3)
        _, _, H, W, M = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W, M)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, M, -1).permute(0, 4, 1, 2, 3).contiguous()
        # print('out', out.shape)

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2, 1), mode='trilinear'))
        #         out = self.attention_gate3(out, t2)
        out = self.layerup3(out)
        out = torch.add(out, t2)

        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2, 1), mode='trilinear'))
        #         out = self.attention_gate4(out, t1)
        out = self.layerup4(out)
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 1), mode='trilinear'))

        return self.final(out)


if __name__ == '__main__':
    #     # summary(a,(4,110,256,256))
    #             input_size = (224, 224)
    a = UNext(1)
    #     # summary(a,(4,110,256,256))
    input_demo = Variable(torch.zeros(10, 1, 224, 224, 4))
    output_demo = a(input_demo)
    b = output_demo.shape[0]
    c = output_demo.shape[1]
    print(output_demo.shape)
    print(b, c)

