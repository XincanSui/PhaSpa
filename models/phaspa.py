import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        # x: (B, C, H, W)
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

# =========================================================================
# 2. ConvNeXt V2 Block
# =========================================================================

class ConvNeXtV2Block(nn.Module):
    """ ConvNeXt V2 Block adapted for NCHW layout """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        
        # 1. Depthwise Convolution (7x7)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        
        # 2. LayerNorm (Channels First)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        # 3. Pointwise Conv (Expansion) -> GELU -> GRN -> Pointwise Conv (Projection)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1) 
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = self.drop_path(x)
        return input + x


class ConvNeXt_Adapter(nn.Module):
    def __init__(self, C_in, C_hid, C_out, depth=1, drop_path=0.0):
        super(ConvNeXt_Adapter, self).__init__()
        self.proj = nn.Conv2d(C_in, C_out, kernel_size=1) if C_in != C_out else nn.Identity()
        
        layers = []
        for i in range(depth):
            layers.append(ConvNeXtV2Block(dim=C_out, drop_path=drop_path))
            
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        x = self.layers(x)
        return x

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class SpatialEncoder(nn.Module):
    """3D SpatialEncoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(SpatialEncoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class SpatialDecoder(nn.Module):
    """3D SpatialDecoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(SpatialDecoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y



import torch
import torch.nn as nn
import torch.fft
from einops import rearrange
import math

class SpatialToPhaseBridge(nn.Module):
    def __init__(self, spatial_dim, phase_dim):
        super().__init__()
        self.compress = nn.Conv2d(spatial_dim, phase_dim, 1)
        
    def forward(self, x_spatial):
        x_phys = self.compress(x_spatial)

        fft_x = torch.fft.rfft2(x_phys, norm='ortho')

        phase = torch.angle(fft_x)

        pos_embed = torch.cat([torch.cos(phase), torch.sin(phase)], dim=1)
        
        return pos_embed


class PhaseToSpatialBridge(nn.Module):
    def __init__(self, phase_dim, spatial_dim):
        super().__init__()

        self.amp_extractor = nn.Sequential(
            nn.Conv2d(spatial_dim, phase_dim, 1),
            nn.InstanceNorm2d(phase_dim, affine=True), 
            nn.ReLU() 
        )

        self.phase_process = nn.Sequential(
            nn.Conv2d(phase_dim * 2, phase_dim * 2, 1),
            nn.GELU() 
        )

        self.spatial_process = nn.Sequential(
            nn.Conv2d(phase_dim, spatial_dim, 3, padding=1),
            nn.GroupNorm(4, spatial_dim),
            nn.SiLU()
        )

        self.gate = nn.Sequential(
            nn.Conv2d(spatial_dim, spatial_dim, 1), 
            nn.Sigmoid()
        )

    def forward(self, x_phase_sc, x_spatial_context):
        """
        x_phase_sc: [B, 2*C_p, H, W_half] 
        x_spatial_context: [B, C_s, H, W] 
        """
        B, _, H, W_half = x_phase_sc.shape
        W = (W_half - 1) * 2 

        feat_spatial_proj = self.amp_extractor(x_spatial_context) # [B, C_p, H, W]

        feat_spatial_fft = torch.fft.rfft2(feat_spatial_proj, norm='ortho') # [B, C_p, H, W_half]

        amplitude_ref = torch.abs(feat_spatial_fft) 

        feat_phase_proc = self.phase_process(x_phase_sc)

        cos_part, sin_part = torch.chunk(feat_phase_proc, 2, dim=1)

        raw_complex = torch.complex(cos_part, sin_part)
        phase_unit = raw_complex / (torch.abs(raw_complex) + 1e-6)

        combined_complex = amplitude_ref * phase_unit

        reconstructed = torch.fft.irfft2(combined_complex, s=(H, W), norm='ortho')
        
        out = self.spatial_process(reconstructed)
        g = self.gate(x_spatial_context)
        
        return out * g

class FreqPhaseMixingBlock(nn.Module):
    """
    处理频域特征 (Cos/Sin 对) 并融合来自空域的指导信息
    """
    def __init__(self, hidden_dim, spatial_dim_for_gate=None):
        super().__init__()
        self.dim = hidden_dim
        in_dim = hidden_dim * 2 

        self.phase_process = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GroupNorm(4, in_dim), 
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1)
        )

        self.ctx_proj = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.SiLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_dim, in_dim, 1)
        )
        
        self.layer_scale = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-2)

    def forward(self, x_phase_sc, x_guidance_sc):
        """
        x_phase_sc: [B, 2*C, H, W_half] 
        x_guidance_sc: [B, 2*C, H, W_half] 
        """

        feat_p = self.phase_process(x_phase_sc)

        feat_ctx = self.ctx_proj(x_guidance_sc)

        combined = torch.cat([feat_p, feat_ctx], dim=1)
        residual = self.fusion(combined)
        
        return x_phase_sc + residual * self.layer_scale

# =========================================================================
# 3. 双流指挥官 (The Orchestrator)
# =========================================================================

class DualStreamTranslator(nn.Module):
    def __init__(self, 
                 simvp_channels, 
                 simvp_hidden,   
                 phase_channels, 
                 N_T, 
                 incep_ker, groups):
        super().__init__()
        self.num_layers = N_T
        
        self.simvp_layers = nn.ModuleList([
            ConvNeXt_Adapter(
                C_in=simvp_channels, 
                C_hid=simvp_channels * 4,
                C_out=simvp_channels,     
                depth=1,                  
                drop_path=0.15          
            )
            for _ in range(N_T)
        ])

        self.phase_layers = nn.ModuleList([
            FreqPhaseMixingBlock(hidden_dim=phase_channels)
            for _ in range(N_T)
        ])

        self.s2p_bridges = nn.ModuleList([
            SpatialToPhaseBridge(spatial_dim=simvp_channels, phase_dim=phase_channels)
            for _ in range(N_T)
        ])

        self.p2s_bridges = nn.ModuleList([
            PhaseToSpatialBridge(phase_dim=phase_channels, spatial_dim=simvp_channels)
            for _ in range(N_T)
        ])

    def forward(self, x_spatial, x_phase_freq):
        """
        x_spatial: [B, C_s, H, W]
        x_phase_freq: [B, 2*C_p, H, W_half] 
        """
        for i in range(self.num_layers):
            x_spatial_new = self.simvp_layers[i](x_spatial)

            guidance_freq = self.s2p_bridges[i](x_spatial_new)

            x_phase_new = self.phase_layers[i](x_phase_freq, guidance_freq)

            feedback_spatial = self.p2s_bridges[i](x_phase_new, x_spatial_new)

            x_spatial = x_spatial_new + feedback_spatial
            x_phase_freq = x_phase_new
            
        return x_spatial, x_phase_freq

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

class DualStreamNet(nn.Module):
    def __init__(self, input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim,
                 n_layers, kernel_size, bias=1):
        super().__init__()
        
        if len(input_shape) == 2:
            self.H, self.W = input_shape
            C = input_dim
        else:
            C, self.H, self.W = input_shape
            
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        
        # --- Configs ---
        self.hid_S = hidden_dim // 2      
        self.N_S = 2         
        self.simvp_latent_C = pre_seq_length * self.hid_S 
        self.phase_C = hidden_dim 

        self.simvp_enc = SpatialEncoder(C, self.hid_S, self.N_S, spatio_kernel=3)

        self.translator = DualStreamTranslator(
            simvp_channels=self.simvp_latent_C,
            simvp_hidden=256,
            phase_channels=self.phase_C, 
            N_T=n_layers,
            incep_ker=[3, 5, 7, 11],
            groups=8
        )

        self.simvp_dec = SpatialDecoder(self.hid_S, C, self.N_S, spatio_kernel=3)

        latent_H, latent_W = self.H // (2**(self.N_S//2)), self.W // (2**(self.N_S//2))
        self.latent_H, self.latent_W = int(latent_H), int(latent_W)

        u_lat = torch.fft.fftfreq(self.latent_H)
        v_lat = torch.fft.rfftfreq(self.latent_W)
        u_lat, v_lat = torch.meshgrid(u_lat, v_lat, indexing='ij')
        uv_lat = torch.stack((u_lat, v_lat), dim=0) # [2, H_lat, W_lat//2+1]
        self.register_buffer('uv_lat', uv_lat) 
        self.pha_conv0 = nn.Conv2d(input_dim * pre_seq_length + 2, self.phase_C, 1)
        self.pha_conv1 = nn.Conv2d(self.phase_C, input_dim * aft_seq_length, 1)
        self.time_adapter = nn.Conv2d(self.simvp_latent_C, aft_seq_length * self.hid_S, 1)

    def pha_norm(self, x): return x / torch.pi
    def pha_unnorm(self, x): return x * torch.pi

    def spectral_pad(self, x_freq, target_shape):
        """
        将 Latent 分辨率的频域特征 (B, T, C, H_low, W_low_half)
        补零填充到 Full 分辨率 (B, T, C, H, W//2+1)
        """
        B, T, C, H_low, W_low_half = x_freq.shape
        H_high, W_high = target_shape
        W_high_half = W_high // 2 + 1
        
        x_high = torch.zeros((B, T, C, H_high, W_high_half), dtype=x_freq.dtype, device=x_freq.device)
        
        w_cut = W_low_half
        h_cut = H_low // 2
        
        x_high[..., :h_cut+1, :w_cut] = x_freq[..., :h_cut+1, :w_cut]
        
        neg_len = H_low - (h_cut + 1)
        if neg_len > 0:
            x_high[..., -neg_len:, :w_cut] = x_freq[..., -neg_len:, :w_cut]
            
        return x_high

    def forward(self, x):
        B, T, C, H, W = x.shape
        x_reshaped = x.reshape(B * T, C, H, W)
        embed, skip = self.simvp_enc(x_reshaped) 
        _, C_latent, H_lat, W_lat = embed.shape

        x_spatial = embed.view(B, T * C_latent, H_lat, W_lat)

        x_small = torch.nn.functional.adaptive_avg_pool2d(x_reshaped, (H_lat, W_lat))
        x_small = x_small.view(B, T, C, H_lat, W_lat)
        
        x_fft = torch.fft.rfft2(x_small, norm='ortho')
        x_phas = self.pha_norm(torch.angle(x_fft)) 
        
        x_phas_flat = rearrange(x_phas, 'b t c h w -> b (t c) h w')
        uv_expanded = self.uv_lat.expand(B, -1, -1, -1)
        x_puv = torch.cat((x_phas_flat, uv_expanded), dim=1)
        
        x_phase_latent_angle = self.pha_conv0(x_puv) # [B, C_p, H_lat, W_lat_half]

        x_phase_freq_sc = torch.cat([
            torch.cos(x_phase_latent_angle * torch.pi), 
            torch.sin(x_phase_latent_angle * torch.pi)
        ], dim=1) # [B, 2*C_p, H_lat, W_lat_half]

        feat_spatial_final, feat_phase_final_sc = self.translator(x_spatial, x_phase_freq_sc)

        hid_trans = self.time_adapter(feat_spatial_final)
        hid_for_dec = hid_trans.reshape(B * self.aft_seq_length, C_latent, H_lat, W_lat)
        
        skip_last = skip.view(B, T, -1, H, W)[:, -1, ...] 
        skip_repeated = skip_last.unsqueeze(1).repeat(1, self.aft_seq_length, 1, 1, 1)
        skip_repeated = skip_repeated.reshape(B * self.aft_seq_length, -1, H, W)

        simvp_feat = hid_for_dec
        for i in range(0, len(self.simvp_dec.dec)-1):
            simvp_feat = self.simvp_dec.dec[i](simvp_feat)
            
        simvp_out = self.simvp_dec.dec[-1](simvp_feat + skip_repeated)
        simvp_pred = self.simvp_dec.readout(simvp_out) 

        xps = rearrange(simvp_pred, '(b t) c h w -> b t c h w', t=self.aft_seq_length)

        cos_final, sin_final = torch.chunk(feat_phase_final_sc, 2, dim=1)
        feat_phase_angle = torch.atan2(sin_final, cos_final) / torch.pi # 归一化回 -1~1

        x_phas_diff = self.pha_conv1(feat_phase_angle)
        x_phas_diff = rearrange(x_phas_diff, 'b (t c) h w -> b t c h w', t=self.aft_seq_length)
        
        x_phas_t = x_phas[:,-1:] + x_phas_diff
        x_phas_t_rad = self.pha_unnorm(x_phas_t)
        x_phas_t_rad[..., 0, 0] = 0.0 
        
        x_amps = torch.abs(x_fft)

        x_phas_full = self.spectral_pad(x_phas_t_rad, (H, W)) # [B, T, C, H, W//2+1]
        x_amps_full = self.spectral_pad(x_amps, (H, W))       # [B, T, C, H, W//2+1]

        return xps, x_phas_full, x_amps_full

class Aligner(nn.Module):
    """
    稳定版幅度对齐器：
    1. 采用残差模式 (xas + diff) 而非生成模式，保证信息流不断。
    2. 采用零初始化，保证初始状态下等同于恒等映射。
    """
    def __init__(self, channels, inter_dim=32):
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(channels * 2, inter_dim, 1),
            nn.GroupNorm(4, inter_dim),
            nn.SiLU()
        )

        self.diff_conv = nn.Sequential(
            nn.Conv2d(inter_dim, inter_dim, kernel_size=7, padding=3, groups=inter_dim),
            nn.GroupNorm(4, inter_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(inter_dim, channels, 1)
        )

        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, inter_dim // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(inter_dim // 2, channels, 1),
            nn.Sigmoid()
        )

        nn.init.constant_(self.diff_conv[-1].weight, 0)
        nn.init.constant_(self.diff_conv[-1].bias, 0)
        
        nn.init.constant_(self.gate_conv[-2].weight, 0)
        nn.init.constant_(self.gate_conv[-2].bias, 0.0) 
        
    def forward(self, xas, xps):
        x_cat = torch.cat([xas, xps], dim=1)
        
        feat = self.entry(x_cat)

        diff = self.diff_conv(feat)
        diff = F.dropout2d(diff, p=0.1, training=True)
        xas_aligned = xas + diff
        
        gate = self.gate_conv(x_cat)
        
        return xas * (1 - gate) + xas_aligned * gate
        
class HighFreqAlphaMixer(nn.Module):
    def __init__(self, input_shape, spec_num, input_dim, hidden_dim, aft_seq_length):
        super().__init__()
        h, w = input_shape
        self.aft_seq_length = aft_seq_length
        self.spec_num = spec_num
        self.hidden_dim = hidden_dim

        # 低频掩码
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[..., :spec_num, :spec_num] = 1.
        spec_mask[..., -spec_num:, :spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)
        
        self.aligner = Aligner(input_dim, inter_dim=hidden_dim)
        
        self.full_spectrum_conv = nn.Sequential(
            ResnetBlock(input_dim, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, 1)
        )

        self.out_mixer = nn.Sequential(
            ResnetBlock(4*input_dim, hidden_dim),
            ResnetBlock(hidden_dim, hidden_dim),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        )

    def forward(self, xas, xps, phas):
        B, T, C, H, W = xas.shape
        xps_fft = torch.fft.rfft2(xps, norm='ortho')
        xps_amps = torch.abs(xps_fft)
        xps_phas = torch.angle(xps_fft) 

        xas_flat = rearrange(xas, 'b t c h w -> (b t) c h w')
        xps_flat = rearrange(xps, 'b t c h w -> (b t) c h w')
        xas_aligned_flat = self.aligner(xas_flat, xps_flat)
        xas_aligned = rearrange(xas_aligned_flat, '(b t) c h w -> b t c h w', t=T)
        
        xas_fft = torch.fft.rfft2(xas_aligned, norm='ortho')
        xas_amps = torch.abs(xas_fft)

        phase_final = xps_phas + (phas - xps_phas) * self.spec_mask
        
        structural_fft = xas_amps * torch.exp(1j * phase_final)
        alpha_structural = torch.fft.irfft2(structural_fft, s=(H, W), norm='ortho')

        alpha_high = self.full_spectrum_conv(xas_aligned_flat)
        alpha_high = rearrange(alpha_high, '(b t) c h w -> b t c h w', t=T)

        xap = torch.cat([xas_aligned, xps, alpha_structural, alpha_high], dim=2)
        xap = rearrange(xap, 'b t c h w -> (b t) c h w')
        xt = self.out_mixer(xap)
        
        return rearrange(xt, '(b t) c h w -> b t c h w', t=T)

class PhaSpa(nn.Module):
    def __init__(self, pre_seq_length, aft_seq_length, input_shape, input_dim,
                 hidden_dim, n_layers, spec_num=20, kernel_size=1, bias=1,
                 pha_weight=0.01, anet_weight=0.1, amp_weight=0.01, aweight_stop_steps=10000):
        super(PhaSpa, self).__init__()

        self.DualStreamNet = DualStreamNet(input_shape, pre_seq_length, aft_seq_length, input_dim, hidden_dim, n_layers, kernel_size, bias)

        self.alphamixer = HighFreqAlphaMixer(input_shape, spec_num, input_dim, hidden_dim // 4, aft_seq_length)
        self.input_shape, self.input_dim = input_shape, input_dim
        self.hidden_dim = hidden_dim
        self.spec_num = spec_num
        self.pha_weight = pha_weight
        self.anet_weight = anet_weight
        self.amp_weight = amp_weight
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.criterion = nn.MSELoss()
        self.itr = 0
        self.aweight_stop_steps = aweight_stop_steps
        self.sampling_changing_rate = self.amp_weight / self.aweight_stop_steps

        h, w = input_shape
        spec_mask = torch.zeros(h, w//2+1)
        spec_mask[..., :spec_num, :spec_num] = 1.
        spec_mask[..., -spec_num:, :spec_num] = 1.
        self.register_buffer('spec_mask', spec_mask)

    def forward(self, x, y=None, cmp_fft_loss=False): # x:[b,t,c,h,w]
        self.itr += 1
        xas = x[:, -1:].repeat(1, self.aft_seq_length, 1, 1, 1)

        xps, x_phas_t, x_amps = self.DualStreamNet(x)
        
        xt_ini = self.alphamixer(xas, xps, x_phas_t)
        xt = xt_ini
        return xt, xps, xas, x_phas_t, x_amps

    def predict(self, frames_in, frames_gt=None, compute_loss=False):
        B = frames_in.shape[0]
        xt, xps, xas, x_phas_t, x_amps = self(frames_in, frames_gt, compute_loss)
        pred = xt
        
        if compute_loss:
            if self.itr < self.aweight_stop_steps:
                self.amp_weight -= self.sampling_changing_rate
            else:
                self.amp_weight  = 0.
            
            loss = 0.
            loss += self.criterion(pred, frames_gt)
            
            frames_fft = torch.fft.rfft2(frames_gt)
            frames_pha = torch.angle(frames_fft)
            frames_abs = torch.abs(frames_fft)
            
            pha_loss = (1 - torch.cos(frames_pha * self.spec_mask - x_phas_t * self.spec_mask)).sum() / (self.spec_mask.sum()*B*self.aft_seq_length*self.input_dim)
            loss += self.pha_weight * pha_loss
            
            xas_fft = torch.fft.rfft2(xas)
            xas_abs = torch.abs(xas_fft)
            pred_fft = torch.fft.rfft2(pred)
            pred_abs = torch.abs(pred_fft)

            eps = 1e-8
            log_gt = torch.log(frames_abs + eps)
            log_pred = torch.log(pred_abs + eps)

            amp_loss = self.criterion(log_pred, log_gt)
            
            loss += self.amp_weight * amp_loss
            anet_loss = self.criterion(xas, frames_gt)
            
            loss_dict = {
                'total_loss': loss, 
                'phase_loss': self.pha_weight * pha_loss,
                'ampli_loss': self.amp_weight * amp_loss, 
                'anet_loss': self.anet_weight * anet_loss
            }
            return pred, loss_dict
        else:
            return pred, None


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros', groupnorm=True):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=kernel_size, padding = kernel_size//2, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out) if groupnorm else nn.BatchNorm2d(dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups = 8, kernel_size=3, padding_mode='zeros'): #'zeros', 'reflect', 'replicate' or 'circular'
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups = groups, kernel_size=kernel_size, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


def Upsample(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, dim_out, 3, padding = 1)
    )

def Downsample(dim, dim_out):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, dim_out, 1)
    )


def get_model(
    img_channels=1,
    dim = 128,
    T_in = 5, 
    T_out = 20,
    input_shape = (128,128),
    n_layers = 5,
    spec_num = 20,
    pha_weight=0.01, 
    anet_weight=0.1,
    amp_weight=0.01,
    aweight_stop_steps=10000,
    **kwargs
):
    model = PhaSpa(pre_seq_length=T_in, aft_seq_length=T_out, input_shape=input_shape, input_dim=img_channels, 
                     hidden_dim=dim, n_layers=n_layers, spec_num=spec_num,
                     pha_weight=pha_weight, anet_weight=anet_weight, amp_weight=amp_weight, aweight_stop_steps=aweight_stop_steps,
                     )
    
    return model

