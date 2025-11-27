import torch
import torch.nn as nn
import torch.nn.functional as F

from xformers.ops import memory_efficient_attention


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads=16):
        super().__init__()
        self.heads = heads
        self.dim_head = query_dim // heads
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # (B, M, H, K)
        q = q.reshape(q.shape[0], -1, self.heads, self.dim_head)
        k = k.reshape(k.shape[0], -1, self.heads, self.dim_head)
        v = v.reshape(v.shape[0], -1, self.heads, self.dim_head)

        # (B, M, H, K)
        out = memory_efficient_attention(q, k, v)
        # (B, H*W, C)
        out = out.reshape(out.shape[0], -1, self.heads * self.dim_head)
        return self.to_out(out)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        return x + identity


class ConditionalUNet2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, action_dim=768, num_heads=16):
        super().__init__()

        # Encoder
        self.inc = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels, base_channels * 2))

        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels * 2, base_channels * 4))

        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels * 4, base_channels * 8))
        self.attn3 = CrossAttention(base_channels * 8, action_dim, num_heads)

        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResBlock(base_channels * 8, base_channels * 16))
        self.attn4 = CrossAttention(base_channels * 16, action_dim, num_heads)

        # Decoder
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ResBlock(base_channels * 16, base_channels * 8)  # *16 because of concat

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock(base_channels * 8, base_channels * 4)  # *8 because of concat

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base_channels * 4, base_channels * 2)  # *4 because of concat

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base_channels * 2, base_channels)  # *2 because of concat

        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x, context):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x4 = self.down3(x3)
        b, c, h, w = x4.shape
        x4_attn = self.attn3(x4.reshape(b, c, -1).transpose(-1, -2), context).transpose(-1, -2).reshape(b, c, h, w)
        x4 = x4 + x4_attn

        x5 = self.down4(x4)
        b, c, h, w = x5.shape
        x5_attn = self.attn4(x5.reshape(b, c, -1).transpose(-1, -2), context).transpose(-1, -2).reshape(b, c, h, w)
        x5 = x5 + x5_attn

        # Decoder
        x = self.up4(x5)
        x = self.dec4(torch.cat([x, x4], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.dec1(torch.cat([x, x1], dim=1))

        return self.outc(x)
