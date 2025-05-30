# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Demystify Mamba in Vision: A Linear Attention Perspective
# Modified by Dongchen Han
# -----------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


def rope(x, shape, base=10000):
    channel_dims, feature_dim = shape[:-1], shape[-1]
    k_max = feature_dim // (2 * len(channel_dims))

    assert feature_dim % k_max == 0

    # angles
    theta_ks = 1 / (base ** (torch.arange(k_max, device=x.device).contiguous() / k_max))
    angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d, device=x.device).contiguous() for d in channel_dims], indexing='ij')], dim=-1)

    # rotation
    rotations_re = torch.cos(angles).unsqueeze(dim=-1).contiguous()
    rotations_im = torch.sin(angles).unsqueeze(dim=-1).contiguous()

    x = x.reshape(*x.shape[:-1], -1, 2).contiguous()
    x_re = x[..., :1]
    x_im = x[..., 1:]
    pe_x = torch.cat([x_re * rotations_re - x_im * rotations_im, x_im * rotations_re + x_re * rotations_im], dim=-1)
    return pe_x.flatten(-2).contiguous()


class LinearAttention(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x, hw_shape=None):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h, w = hw_shape
        assert h * w == n
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3).contiguous()
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = rope(q.reshape(b, h, w, c).contiguous(), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k_rope = rope(k.reshape(b, h, w, c).contiguous(), (h, w, c)).reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1).contiguous() + 1e-6)
        kv = (k_rope.transpose(-2, -1).contiguous() * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c).contiguous()
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c).contiguous()

        return x


class FocusedLinearAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, proj_drop=0.,
                 focusing_factor=3, kernel_size=5):

        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.focusing_factor = focusing_factor
        self.qk_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        print('Linear Attention window{} f{} kernel{} dim{} num_heads{}'.format(window_size, focusing_factor, kernel_size, dim, num_heads))
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

    def forward(self, x, hw_shape):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
        """
        img_H, img_W = hw_shape

        B, N, C = x.shape      
        qk = self.qk_linear(x).reshape(B, N, 2, C).permute(2, 0, 1, 3).contiguous()
        q, k = qk.unbind(0)
        v = x

        q = rope(q.reshape(B, img_H, img_W, C).contiguous(), (img_H, img_W, C)).reshape(B, N, C).contiguous()
        k = rope(k.reshape(B, img_H, img_W, C).contiguous(), (img_H, img_W, C)).reshape(B, N, C).contiguous()
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        focusing_factor = self.focusing_factor
        scale = nn.Softplus()(self.scale)
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1).contiguous() + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C).contiguous()
        v = v.reshape(B * self.num_heads, img_H, img_W, -1).permute(0, 3, 1, 2).contiguous()
        x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1).contiguous()

        return x



class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()
    

class MLLABlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fcla_window_size=7, fcla_proj_drop=0., fcla_focusing_factor=3, fcla_kernel_size=5, atten_type="L",
                 **kwargs):
        super().__init__() 
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        if atten_type == "L":
            self.attn = LinearAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias)
        else:
            self.attn = FocusedLinearAttention(dim=dim, window_size=fcla_window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                            proj_drop=fcla_proj_drop, focusing_factor=fcla_focusing_factor, kernel_size=fcla_kernel_size
                                            )
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, hw_shape):
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()).flatten(2).permute(0, 2, 1).contiguous()
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C).contiguous()
        x = self.act(self.dwc(x.permute(0, 3, 1, 2).contiguous())).permute(0, 2, 3, 1).view(B, L, C).contiguous()

        # Linear Attention
        x = self.attn(x, hw_shape)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()).flatten(2).permute(0, 2, 1).contiguous()

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(int(out_channels * ratio), int(out_channels * ratio), kernel_size=3, stride=2, padding=1, groups=int(out_channels * ratio), norm=None),
            ConvLayer(int(out_channels * ratio), out_channels, kernel_size=1, act_func=None)
        )

    def forward(self, x, input_size):
        """
        x: B, H*W, C
        """
        H, W = input_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.reshape(B, H, W, C).contiguous()

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x = self.conv(x.permute(0, 3, 1, 2).contiguous())
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).permute(0, 2, 1).contiguous()

        return x, out_size


class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, 
                 use_dmca=False, dmca_dim=96, dmca_cur_sdim=96, dmca_ssm_ratio=2, dmca_sr_ratio=1, dmca_heads=4,
                 fcla_window_size=7, fcla_proj_drop=0., fcla_focusing_factor=3, fcla_kernel_size=5, atten_type="L"
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # DMCA op
        self.use_dmca = use_dmca

        # build blocks
        self.blocks = nn.ModuleList([
            MLLABlock(dim=dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      fcla_window_size=fcla_window_size, fcla_proj_drop=fcla_proj_drop, fcla_focusing_factor=fcla_focusing_factor, fcla_kernel_size=fcla_kernel_size, atten_type=atten_type
                      )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, hw_shape)
            else:
                x = blk(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


class Stem(nn.Module):
    r""" Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            ConvLayer(embed_dim // 2, embed_dim // 2, kernel_size=3, stride=1, padding=1, bias=False, act_func=None)
        )
        self.conv3 = nn.Sequential(
            ConvLayer(embed_dim // 2, embed_dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            ConvLayer(embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None)
        )

    def forward(self, x):
        # padding
        _, _, H, W = x.size()
        
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2).contiguous()  # B Ph*Pw C
        return x, out_size



class MLLABackbone(nn.Module):
    r""" MLLA
        A PyTorch impl of : `Demystify Mamba in Vision: A Linear Attention Perspective`

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each MLLA layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.1,
                 out_indices=(0, 1, 2, 3), init_cfg=None,
                 norm_layer=nn.LayerNorm, ape=False, use_checkpoint=False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            stages_norm_layer = norm_layer(self.num_features[i])
            stages_norm_layer_name = f'norm{i}'
            self.add_module(stages_norm_layer_name, stages_norm_layer)
        self.mlp_ratio = mlp_ratio

        self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # dmca params
        use_dmca = False        # disable dmca
        dmca_dim_list = [self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8]
        dmca_cur_sdim_list = [self.embed_dim * 4, self.embed_dim * 8, self.embed_dim * 16]
        dmca_ssm_ratio_list = [2, 2, 2]
        dmca_sr_ratio_list = [4, 2, 1]
        dmca_heads_list = [8, 16, 32]

        # focus linear attention
        fcla_window_size_list = [self.img_size//4, self.img_size//8, self.img_size//16, self.img_size//32] 
        fcla_proj_drop_list = [0., 0., 0., 0.]
        fcla_focusing_factor_list = [3, 3, 3, 3] 
        fcla_kernel_size_list = [5, 5, 5, 5]
        atten_type_list = ["F", "F", "F", "F"]

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer < 1:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                        patches_resolution[1] // (2 ** i_layer)),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, drop=drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint,
                                    fcla_window_size=fcla_window_size_list[i_layer],
                                    fcla_proj_drop=fcla_proj_drop_list[i_layer], 
                                    fcla_focusing_factor=fcla_focusing_factor_list[i_layer], 
                                    fcla_kernel_size=fcla_kernel_size_list[i_layer],
                                    atten_type=atten_type_list[i_layer]
                                )
            else:
                layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                    input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                      patches_resolution[1] // (2 ** i_layer)),
                                    depth=depths[i_layer],
                                    num_heads=num_heads[i_layer],
                                    mlp_ratio=self.mlp_ratio,
                                    qkv_bias=qkv_bias, drop=drop_rate,
                                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                    norm_layer=norm_layer,
                                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint,
                                    use_dmca=use_dmca,
                                    dmca_dim=dmca_dim_list[i_layer - 1],
                                    dmca_cur_sdim=dmca_cur_sdim_list[i_layer - 1],
                                    dmca_ssm_ratio=dmca_ssm_ratio_list[i_layer - 1],
                                    dmca_sr_ratio=dmca_sr_ratio_list[i_layer - 1],
                                    dmca_heads=dmca_heads_list[i_layer - 1],
                                    fcla_window_size=fcla_window_size_list[i_layer],
                                    fcla_proj_drop=fcla_proj_drop_list[i_layer], 
                                    fcla_focusing_factor=fcla_focusing_factor_list[i_layer], 
                                    fcla_kernel_size=fcla_kernel_size_list[i_layer],
                                    atten_type=atten_type_list[i_layer]
                                    )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x, hw_shape, out, out_hw_shape = layer(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}') 
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).contiguous().permute(0, 3, 1, 2).contiguous()
                outs.append(out)

        return outs

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops
