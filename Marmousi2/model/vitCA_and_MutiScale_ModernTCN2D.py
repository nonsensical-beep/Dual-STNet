import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm2d(channels)


def conv_bn_2d(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        B, C, H, W = x.size()
        x_h = self.pool_h(x)  # 压缩水平方向: (B, C, H, W) --> (B, C, H, 1)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 压缩垂直方向: (B, C, H, W) --> (B, C, 1, W) --> (B,C,W,1)

        # 坐标注意力生成
        y = torch.cat([x_h, x_w], dim=2)  # 拼接水平和垂直方向的向量: (B,C,H+W,1)
        y = self.conv1(y)  # 通过Conv进行变换,并降维: (B,C,H+W,1)--> (B,d,H+W,1)
        y = self.bn1(y)  # BatchNorm操作: (B,d,H+W,1)
        y = self.relu(y)  # Relu操作: (B,d,H+W,1)

        x_h, x_w = torch.split(y, [H, W], dim=2)  # 沿着空间方向重新分割为两部分: (B,d,H+W,1)--> x_h:(B,d,H,1); x_w:(B,d,W,1)
        x_w = x_w.permute(0, 1, 3, 2)  # x_w: (B,d,W,1)--> (B,d,1,W)

        a_h = self.conv_h(x_h).sigmoid()  # 恢复与输入相同的通道数,并生成垂直方向的权重: (B,d,H,1)-->(B,C,H,1)
        a_w = self.conv_w(x_w).sigmoid()  # 恢复与输入相同的通道数,并生成水平方向的权重: (B,d,1,W)-->(B,C,1,W)

        out = identity * a_w * a_h  # 将垂直、水平方向权重应用于输入,从而反映感兴趣的对象是否存在于相应的行和列中: (B,C,H,W) * (B,C,1,W) * (B,C,H,1) = (B,C,H,W)

        return out


class MSCNN2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=12):
        super(MSCNN2D, self).__init__()

        self.gelu = nn.GELU()

        # 多尺度卷积分支
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=1 * (5 - 1) // 2,
                      dilation=1),
            nn.GroupNorm(1, 12),
            self.gelu
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=3 * (5 - 1) // 2,
                      dilation=3),
            nn.GroupNorm(1, 12),
            self.gelu
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=5 * (5 - 1) // 2,
                      dilation=5),
            nn.GroupNorm(1, 12),
            self.gelu
        )

        # 融合卷积
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=32, kernel_size=5, padding=2),
            nn.GroupNorm(1, 32),
            self.gelu,

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.GroupNorm(1, 32),
            self.gelu,

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
            nn.GroupNorm(1, 32),
            self.gelu,
        )

        # 输出层 (逐点映射 16→16)
        self.out = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):  # x: [B, 3, H, W]
        cnn_out1 = self.cnn1(x)  # [B, 8, H, W]
        cnn_out2 = self.cnn2(x)  # [B, 8, H, W]
        cnn_out3 = self.cnn3(x)  # [B, 8, H, W]

        cnn_out = torch.cat([cnn_out1, cnn_out2, cnn_out3], dim=1)  # [B, 24, H, W]
        cnn_out = self.cnn(cnn_out)  # [B, 16, H, W]

        out = self.out(cnn_out)  # [B, 16, H, W]
        out = self.gelu(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)  # Linear 层用 Xavier
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0):
        super(ConvDownsample, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        pad_h = 0 if H % 2 == 0 else 1
        if pad_h:
            x = F.pad(x, (0, 0, 0, pad_h))

        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=(2, 2), mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, target_size=None):
        if target_size is not None:
            x = F.interpolate(x.cpu(), size=target_size, mode=self.mode, align_corners=False).to(x.device)
        else:
            x = F.interpolate(x.cpu(), scale_factor=self.scale_factor, mode=self.mode, align_corners=False).to(x.device)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ===== Patch Embedding =====
class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=64, patch_size=(16, 2)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x, H, W, pad_h, pad_w


# ===== Multi-Head Self-Attention (with dropout) =====
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: [B, heads, N, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ===== MLP =====
class MLP(nn.Module):
    def __init__(self, in_features, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        hidden = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DualBranchViTBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop=0.1, ln_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)

        self.attn = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)

        self.cattn = CoordAtt(inp=dim, oup=dim, reduction=32)

        self.alpha = nn.Parameter(torch.tensor(0.5))

        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x, H, W):
        """
        x: (B, N, C)  <-- ViT 的标准输入格式 (Flattened patches)
        H, W: 当前特征图的高度和宽度 (用于 CA 重塑)
        """
        B, N, C = x.shape

        x_norm = self.norm1(x)

        out_msa = self.attn(x_norm)

        x_img = x_norm.transpose(1, 2).reshape(B, C, H, W)
        out_ca = self.cattn(x_img)
        out_ca = out_ca.flatten(2).transpose(1, 2)

        x = x + (self.alpha * out_msa + (1 - self.alpha) * out_ca)

        x = x + self.mlp(self.norm2(x))

        return x

class ViTEncoder(nn.Module):
    def __init__(self, in_chans=32, embed_dim=64, patch_size=(2, 3), num_vit_blocks=3):
        super().__init__()

        # Stage 1
        self.patch_embed_1 = PatchEmbed(in_chans, embed_dim, patch_size)
        self.blocks_1 = nn.ModuleList([DualBranchViTBlock(embed_dim) for _ in range(num_vit_blocks)])

        # Stage 2
        self.patch_embed_2 = PatchEmbed(embed_dim, embed_dim * 2, patch_size=(2, 1))
        self.blocks_2 = nn.ModuleList([DualBranchViTBlock(embed_dim * 2) for _ in range(num_vit_blocks)])

        # Stage 3
        self.patch_embed_3 = PatchEmbed(embed_dim * 2, embed_dim * 4, patch_size=(2, 1))
        self.blocks_3 = nn.ModuleList([DualBranchViTBlock(embed_dim * 4) for _ in range(num_vit_blocks)])

    def forward(self, x):
        shapes = []
        skips = []
        B = x.shape[0]
        shapes.append(x.shape[2:])
        skips.append(x)

        x1, H1, W1, pad_h1, pad_w1 = self.patch_embed_1(x)

        for blk in self.blocks_1:
            x1 = blk(x1, H1, W1)

        x1_reshaped = x1.transpose(1, 2).reshape(B, -1, H1, W1)
        shapes.append(x1_reshaped.shape[2:])
        skips.append(x1_reshaped)

        x2, H2, W2, pad_h2, pad_w2 = self.patch_embed_2(x1_reshaped)

        for blk in self.blocks_2:
            x2 = blk(x2, H2, W2)

        x2_reshaped = x2.transpose(1, 2).reshape(B, -1, H2, W2)
        shapes.append(x2_reshaped.shape[2:])
        skips.append(x2_reshaped)

        x3, H3, W3, pad_h3, pad_w3 = self.patch_embed_3(x2_reshaped)

        for blk in self.blocks_3:
            x3 = blk(x3, H3, W3)

        x3_reshaped = x3.transpose(1, 2).reshape(B, -1, H3, W3)
        shapes.append(x3_reshaped.shape[2:])
        skips.append(x3_reshaped)

        return x3_reshaped, shapes, skips


class ReparamLargeKernelConv2D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super(ReparamLargeKernelConv2D, self).__init__()

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
            padding = (kernel_size // 2, kernel_size // 2)
        else:
            self.kernel_size = kernel_size
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        if isinstance(small_kernel, int):
            small_padding = (small_kernel // 2, small_kernel // 2)
        else:
            small_padding = (small_kernel[0] // 2, small_kernel[1] // 2)

        self.small_kernel = small_kernel

        if small_kernel_merged:
            self.lkb_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=self.kernel_size, stride=stride, padding=padding, dilation=1,
                                         groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn_2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=self.kernel_size, stride=stride, padding=padding, dilation=1,
                                         groups=groups, bias=False)
            if small_kernel is not None:
                self.small_conv = conv_bn_2d(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=small_kernel, stride=stride, padding=small_padding,
                                             groups=groups, dilation=1, bias=False)

    def forward(self, inputs):
        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out


class MultiScaleReparamBlock(nn.Module):
    def __init__(self, channels, kernel_sizes, stride, groups, small_kernel, small_kernel_merged=False):
        super(MultiScaleReparamBlock, self).__init__()
        self.branches = nn.ModuleList()
        for k_size in kernel_sizes:
            branch = ReparamLargeKernelConv2D(
                in_channels=channels,
                out_channels=channels,
                kernel_size=k_size,
                stride=stride,
                groups=groups,
                small_kernel=small_kernel,
                small_kernel_merged=small_kernel_merged
            )
            self.branches.append(branch)

    def forward(self, x):
        out = 0
        for branch in self.branches:
            out += branch(x)
        return out


class Block2D(nn.Module):
    def __init__(self, kernel_sizes, small_size, dmodel, dff, drop=0.1):
        super(Block2D, self).__init__()

        self.dw = MultiScaleReparamBlock(channels=dmodel, kernel_sizes=kernel_sizes, stride=1, groups=dmodel,
                                         small_kernel=small_size, small_kernel_merged=False)

        self.norm = nn.BatchNorm2d(dmodel)

        self.pw1 = nn.Conv2d(in_channels=dmodel, out_channels=dff, kernel_size=1, stride=1, padding=0)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(in_channels=dff, out_channels=dmodel, kernel_size=1, stride=1, padding=0)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.dw(x)
        x = self.norm(x)

        x = self.pw1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.pw2(x)
        x = self.drop2(x)
        x = input + x
        return x


class Stage2D(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, kernel_sizes, small_size, dmodel, drop=0.1):
        super(Stage2D, self).__init__()
        d_ffn = int(dmodel * ffn_ratio)
        blks = []
        for i in range(num_blocks):
            blk = Block2D(kernel_sizes=kernel_sizes, small_size=small_size, dmodel=dmodel, dff=d_ffn, drop=drop)
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCN2DEncoder(nn.Module):
    def __init__(self, ffn_ratio=2, base_dim=32, kernel_sizes=[(51, 1), (21, 1), (7, 1)], small_size=(5, 3),
                 num_blocks=2, drop=0.1):
        super(ModernTCN2DEncoder, self).__init__()
        self.stage1 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=kernel_sizes,
                              small_size=small_size, dmodel=base_dim, drop=drop)
        self.down1 = ConvDownsample(in_channels=base_dim, out_channels=base_dim * 2, kernel_size=(3, 3), stride=(2, 3),
                                    padding=(1, 0))

        self.stage2 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=kernel_sizes,
                              small_size=small_size, dmodel=base_dim * 2, drop=drop)
        self.down2 = ConvDownsample(in_channels=base_dim * 2, out_channels=base_dim * 4, kernel_size=(3, 1),
                                    stride=(2, 1), padding=(1, 0))

        self.stage3 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=kernel_sizes,
                              small_size=small_size, dmodel=base_dim * 4, drop=drop)
        self.down3 = ConvDownsample(in_channels=base_dim * 4, out_channels=base_dim * 8, kernel_size=(3, 1),
                                    stride=(2, 1), padding=(1, 0))

    def forward(self, x):
        shapes = []  # 记录尺寸
        skips = []  # 用于跳跃连接

        shapes.append(x.shape[2:])
        skips.append(x)
        x1 = self.stage1(x)
        x1 = self.down1(x1)
        shapes.append(x1.shape[2:])
        skips.append(x1)

        x2 = self.stage2(x1)
        x2 = self.down2(x2)
        shapes.append(x2.shape[2:])
        skips.append(x2)

        x3 = self.stage3(x2)
        x3 = self.down3(x3)
        shapes.append(x3.shape[2:])
        skips.append(x3)

        return x3, shapes, skips


class ModernTCN2DDecoder(nn.Module):
    def __init__(self, ffn_ratio=2, base_dim=32, kernel_sizes=[(51, 1), (21, 1), (7, 1)], small_size=(5, 3),
                 num_blocks=2, drop=0.1):
        super(ModernTCN2DDecoder, self).__init__()
        self.up1 = ConvUpsample(in_channels=base_dim * 8, out_channels=base_dim * 4, kernel_size=(3, 3),
                                scale_factor=(2, 1))
        self.stage1 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=[(51, 1), (21, 1), (7, 1)],
                              small_size=small_size, dmodel=base_dim * 4, drop=drop)

        self.up2 = ConvUpsample(in_channels=base_dim * 4, out_channels=base_dim * 2, kernel_size=(3, 3),
                                scale_factor=(2, 1))
        self.stage2 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=[(51, 1), (21, 1), (7, 1)],
                              small_size=small_size, dmodel=base_dim * 2, drop=drop)

        self.up3 = ConvUpsample(in_channels=base_dim * 2, out_channels=base_dim, kernel_size=(3, 3),
                                scale_factor=(2, 3))
        self.stage3 = Stage2D(ffn_ratio=ffn_ratio, num_blocks=num_blocks, kernel_sizes=[(51, 1), (21, 1), (7, 1)],
                              small_size=small_size, dmodel=base_dim, drop=drop)

        self.final_conv = nn.Conv2d(in_channels=base_dim, out_channels=base_dim, kernel_size=(1, 1))

    def forward(self, x, shapes, skips, vit_skips):
        x = x + vit_skips[-1]

        x = self.up1(x, target_size=shapes[2])
        x = x + skips[2] + vit_skips[2]
        x = self.stage1(x)

        x = self.up2(x, target_size=shapes[1])
        x = x + skips[1] + vit_skips[1]
        x = self.stage2(x)

        x = self.up3(x, target_size=shapes[0])
        x = x + skips[0]
        x = self.stage3(x)

        out = self.final_conv(x)
        return out


class inverse_model(nn.Module):
    def __init__(self, in_chans=6, ffn_ratio=2, base_dim=32, kernel_sizes=[(51, 1), (21, 1), (7, 1)],
                 small_size=(5, 3), num_blocks=2, drop=0.2, embed_dim=64, patch_size=(2, 3)):
        super(inverse_model, self).__init__()
        self.mscnn = MSCNN2D(in_channels=in_chans, out_channels=in_chans * 2)
        self.vit = ViTEncoder(in_chans=base_dim, embed_dim=embed_dim, patch_size=patch_size, num_vit_blocks=3)
        self.tcn = ModernTCN2DEncoder(ffn_ratio=ffn_ratio, base_dim=base_dim, kernel_sizes=kernel_sizes,
                                      small_size=small_size, num_blocks=num_blocks, drop=drop)
        self.decoder_vs = ModernTCN2DDecoder(ffn_ratio=ffn_ratio, base_dim=base_dim, kernel_sizes=kernel_sizes,
                                             small_size=small_size, num_blocks=num_blocks, drop=drop)
        self.decoder_vp = ModernTCN2DDecoder(ffn_ratio=ffn_ratio, base_dim=base_dim, kernel_sizes=kernel_sizes,
                                             small_size=small_size, num_blocks=num_blocks, drop=drop)
        self.decoder_den = ModernTCN2DDecoder(ffn_ratio=ffn_ratio, base_dim=base_dim, kernel_sizes=kernel_sizes,
                                              small_size=small_size, num_blocks=num_blocks, drop=drop)
        self.groupnorm = nn.GroupNorm(num_channels=base_dim, num_groups=1)

        self.conv_vs = nn.Sequential(
            nn.Conv2d(in_channels=base_dim, out_channels=20, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=20, num_groups=1),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=10, num_groups=1),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(1, 3), padding=(0, 0)))

        self.conv_vp = nn.Sequential(
            nn.Conv2d(in_channels=base_dim, out_channels=20, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=20, num_groups=1),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=10, num_groups=1),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(1, 3), padding=(0, 0)))

        self.conv_den = nn.Sequential(
            nn.Conv2d(in_channels=base_dim, out_channels=20, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=20, num_groups=1),
            nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.GroupNorm(num_channels=10, num_groups=1),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=(1, 3), padding=(0, 0)))

    def forward(self, x, init):
        x = torch.cat((x, init), dim=1)
        x = self.mscnn(x)
        vit_out, vit_shapes, vit_skips = self.vit(x)
        tcn_out, tcn_shapes, tcn_skips = self.tcn(x)
        vs = self.decoder_vs(tcn_out, tcn_shapes, tcn_skips, vit_skips)
        vs = self.groupnorm(vs)
        vs = self.conv_vs(vs).squeeze(-1)

        vp = self.decoder_vp(tcn_out, tcn_shapes, tcn_skips, vit_skips)
        vp = self.groupnorm(vp)
        vp = self.conv_vp(vp).squeeze(-1)

        den = self.decoder_den(tcn_out, tcn_shapes, tcn_skips, vit_skips)
        den = self.groupnorm(den)
        den = self.conv_den(den).squeeze(-1)

        return vs, vp, den


class forward_model(nn.Module):
    def __init__(self, resolution_ratio=4, nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.cnn1 = nn.Conv1d(in_channels=3, out_channels=4, kernel_size=9, padding=4)
        self.cnn2 = nn.Conv1d(in_channels=4, out_channels=4, kernel_size=7, padding=3)
        self.cnn3 = nn.Conv1d(in_channels=4, out_channels=3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        return x


# ===== 测试 =====
if __name__ == "__main__":
    dummy_input = torch.randn(8, 3, 1501, 3)  # B=2, C=3, H=128, W=64
    ini_input = torch.randn(8, 3, 1501, 3)
    model = inverse_model()
    vs, vp, den = model(dummy_input, ini_input)
    print(vs.shape)
    print(vp.shape)
    print(den.shape)
