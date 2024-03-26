import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal

class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = x * y.expand(x.size())
        return y
    
class GroupBatchnorm3d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 8,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm3d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, D, H, W = x.size()  # 3D输入的维度
        x = x.view(N, self.group_num, -1, D, H, W)  # 将数据重塑为(N, group_num, -1, D, H, W)

        # 计算每个通道组内数据的均值和标准差
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)

        # 归一化操作
        x = (x - mean) / (std + self.eps)

        # 将数据重新形状为与输入相同的形状
        x = x.view(N, C, D, H, W)

        # 应用权重和偏置
        return x * self.weight + self.bias


class SRM(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 8,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()

        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm3d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        # w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        # w_gamma = w_gamma.view(1, -1, 1, 1)
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1, 1)

        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reorganization(x_1, x_2)
        return x

    def reorganization(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)

class SRblock(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRM(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)

    def forward(self, x):
        x = self.SRU(x)
        return x
    
class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    
    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        x = x[:,:,1:-1,1:-1,1:-1]
        return self.actout(x)

class DeconvBlock(nn.Module):
    def __init__(self, dec_channels, skip_channels):
        super(DeconvBlock, self).__init__()
        self.upconv = UpConvBlock(dec_channels, skip_channels)
        self.conv = nn.Sequential(
            ConvInsBlock(2*skip_channels, skip_channels),
            ConvInsBlock(skip_channels, skip_channels)
        )
    def forward(self, dec, skip):
        dec = self.upconv(dec)
        out = self.conv(torch.cat([dec, skip], dim=1))
        return out

class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c),
            # SRblock(2 * c)
            SEblock(2 * c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c),
            # SRblock(4 * c)
            SEblock(4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c),
            # SRblock(8 * c)
            SEblock(8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c),
            # SRblock(16 * c)
            SEblock(16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c),
            # SRblock(32 * c)
            SEblock(32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4

class ProjectionLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat

class HFFM(nn.Module):
    def __init__(self, in_channels, channels):
        super(HFFM, self).__init__()

        c = channels
        self.num_fields = in_channels // 3

        self.conv = nn.Sequential(
            ConvInsBlock(in_channels, channels, 3, 1),
            ConvInsBlock(channels, channels, 3, 1),
            nn.Conv3d(channels, self.num_fields, 3, 1, 1),
            nn.Softmax(dim=1)
        )

        self.upsample = nn.Upsample(
                scale_factor=2,
                mode='trilinear',
                align_corners=True
            )

    def forward(self, x):

        x = self.upsample(x)
        weight = self.conv(x)

        weighted_field = 0

        for i in range(self.num_fields):
            w = x[:, 3*i: 3*(i+1)]
            weight_map = weight[:, i:(i+1)]
            weighted_field = weighted_field + w*weight_map

        return 2*weighted_field


class LSPTransformer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()


        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def makeV(self, N):
        # v.shape: (1, N, self.num_heads, self.kernel_size**3, 3)
        v = self.grid.reshape(self.kernel_size**3, 3).unsqueeze(0).unsqueeze(0).repeat(N, self.num_heads, 1, 1).unsqueeze(0)
        return v

    def apply_pb(self, attn, N):
        # attn: B, N, self.num_heads, 1, tokens = (3x3x3)
        bias_idx = torch.arange(self.rpb_size**3).unsqueeze(-1).repeat(N, 1)
        return attn + self.rpb.flatten(1,3)[:, bias_idx].reshape(self.num_heads, N, 1, self.rpb_size**3).transpose(0,1)

    def forward(self, q, k):

        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size ** 3)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # 1, N, heads, 1, head_dim
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        k = k.flatten(0, 1)  # C, H+2, W+2, T+2
        k = k.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).permute(0, 4, 5, 6, 1, 2, 3)  # C, 3, 3, 3, H, W, T
        k = k.reshape(B, self.num_heads, C // self.num_heads, num_tokens, N)  # memory boom
        k = k.permute(0, 4, 1, 3, 2)  # (B, N, heads, num_tokens, head_dim)

        attn = (q @ k.transpose(-2, -1))  # =>B x N x heads x 1 x num_tokens
        if self.use_rpb:
            attn = self.apply_pb(attn, N)
        attn = attn.softmax(dim=-1)

        v = self.makeV(N)  # B, N, heads, num_tokens, 3
        x = (attn @ v)  # B x N x heads x 1 x 3
        x = x.reshape(B, H, W, T, self.num_heads*3).permute(0, 4, 1, 2, 3)

        return x


class MDH(nn.Module):
    def __init__(self,
                 inshape=(160,192,160),
                 in_channel=1,
                 channels=4,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=None):
        super(MDH, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.projblock1 = ProjectionLayer(2*c, dim=head_dim*num_heads[4])
        self.lspt_stage1 = LSPTransformer(head_dim*num_heads[4], num_heads[4], qk_scale=scale)

        self.projblock2 = ProjectionLayer(4*c, dim=head_dim*num_heads[3])
        self.lspt_stage2 = LSPTransformer(head_dim*num_heads[3], num_heads[3], qk_scale=scale)

        self.projblock3 = ProjectionLayer(8*c, dim=head_dim*num_heads[2])
        self.lspt_stage3 = LSPTransformer(head_dim*num_heads[2], num_heads[2], qk_scale=scale)
        self.HFFM3 = HFFM(3 * num_heads[2], 3 * num_heads[2] * 2)

        self.projblock4 = ProjectionLayer(16*c, dim=head_dim*num_heads[1])
        self.lspt_stage4 = LSPTransformer(head_dim*num_heads[1], num_heads[1], qk_scale=scale)
        self.HFFM4 = HFFM(3 * num_heads[1], 3 * num_heads[1] * 2)

        self.projblock5 = ProjectionLayer(32*c, dim=head_dim*num_heads[0])
        self.lspt_stage5 = LSPTransformer(head_dim*num_heads[0], num_heads[0], qk_scale=scale)
        self.HFFM5 = HFFM(3*num_heads[0], 3*num_heads[0]*2)

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

    def forward(self, moving, fixed):

        # dual-stream encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        q5, k5 = self.projblock5(F5), self.projblock5(M5)
        w = self.lspt_stage5(q5, k5)
        w = self.HFFM5(w)
        flow = w

        M4 = self.transformer[3](M4, flow)
        q4,k4 = self.projblock4(F4), self.projblock4(M4)
        w=self.lspt_stage4(q4, k4)
        w = self.HFFM4(w)
        flow = self.transformer[2](self.upsample_trilin(2*flow), w)+w

        M3 = self.transformer[2](M3, flow)
        q3, k3 = self.projblock3(F3), self.projblock3(M3)
        w = self.lspt_stage3(q3, k3)
        w = self.HFFM3(w)
        flow = self.transformer[1](self.upsample_trilin(2 * flow), w) + w

        M2 = self.transformer[1](M2, flow)
        q2,k2 = self.projblock2(F2), self.projblock2(M2)
        w=self.lspt_stage2(q2, k2)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](M1, flow)
        q1, k1 = self.projblock1(F1), self.projblock1(M1)
        w=self.lspt_stage1(q1, k1)
        flow = self.transformer[0](flow, w)+w

        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow

if __name__ == '__main__':
    inshape = (1, 1, 80, 96, 80)
    model = MDH(inshape[2:]).cuda(2)
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(2), B.cuda(2))
