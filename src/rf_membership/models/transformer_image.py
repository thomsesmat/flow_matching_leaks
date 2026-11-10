# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT: https://github.com/facebookresearch/DiT
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp
from rf_membership.models.transformer import TimestepEmbedder, modulate




#################################################################################
#               Embedding Layers for Timesteps                                  #
#################################################################################



#################################################################################
#                          Patch Embedding                                      #
#################################################################################

class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding (inspiré de DiT).
    Convertit (B, C, H, W) → (B, num_patches, hidden_size)
    """
    def __init__(self, patch_size=2, in_channels=4, hidden_size=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, hidden_size,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)          # (B, hidden, H//p, W//p)
        x = x.flatten(2)          # (B, hidden, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, hidden)
        return x


class FinalLayer(nn.Module):
    """
    Final layer pour images avec adaLN + projection vers patch.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#                     Rotary Positional Embedding (RoPE)                        #
#################################################################################

def build_rotary_frequencies(dim, max_seq_len=8192, base=10000, device=None):
    """Construit les fréquences pour RoPE."""
    assert dim % 2 == 0, "La dimension doit être paire pour RoPE"
    
    half_dim = dim // 2
    freq_inv = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freq_inv)
    
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Applique RoPE à un tenseur."""
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    seq_len = x_reshaped.shape[-3]
    
    freqs_cos = freqs_cos[:seq_len, :]
    freqs_sin = freqs_sin[:seq_len, :]
    
    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]
    
    x_rotated_real = x_real * freqs_cos - x_imag * freqs_sin
    x_rotated_imag = x_real * freqs_sin + x_imag * freqs_cos
    
    x_out = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
    x_out = x_out.flatten(-2)
    
    return x_out.type_as(x)


class FlashAttentionWithRoPE(nn.Module):
    """Attention multi-têtes avec Flash Attention ET RoPE."""
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=True,
        dropout=0.0,
        max_seq_len=256,
        rope_base=10000
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        freqs_cos, freqs_sin = build_rotary_frequencies(
            self.head_dim, max_seq_len=max_seq_len, base=rope_base
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, x, is_causal=False):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = apply_rotary_emb(q, self.freqs_cos, self.freqs_sin)
        k = apply_rotary_emb(k, self.freqs_cos, self.freqs_sin)
        
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out


#################################################################################
#                              Transformer Block                                #
#################################################################################

class TBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning."""
    
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, pos_enc="RoPE", max_seq_len=256, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if pos_enc == "RoPE":
            self.attn = FlashAttentionWithRoPE(
                dim=hidden_size, num_heads=num_heads, qkv_bias=True,
                dropout=0.0, max_seq_len=max_seq_len, rope_base=10000
            )
        else:  # SinPE ou woPE
            self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


#################################################################################
#                         2D Positional Embedding                               #
#################################################################################

def get_2d_sincos_pos_embed(hidden_size, grid_size):
    """Génère un positional embedding 2D sin-cos."""
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing='ij')
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(hidden_size, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(hidden_size, grid):
    emb_h = get_1d_sincos_pos_embed_from_grid(hidden_size // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(hidden_size // 2, grid[1])
    return torch.cat([emb_h, emb_w], dim=1)


def get_1d_sincos_pos_embed_from_grid(hidden_size, pos):
    omega = torch.arange(hidden_size // 2, dtype=torch.float32)
    omega = 1. / (10000 ** (omega / (hidden_size / 2.)))
    
    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)
    
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


#################################################################################
#                              Image Transformer                                #
#################################################################################

class ImageTransformer(nn.Module):
    """
    Transformer pour images latentes (style DiT).
    
    Args:
        in_channels: nombre de canaux du latent (4 pour SD VAE)
        hidden_size: dimension cachée du transformer
        depth: nombre de blocs
        num_heads: nombre de têtes d'attention
        mlp_ratio: ratio pour le MLP
        pos_enc: "RoPE", "SinPE", ou "woPE"
        patch_size: taille des patches (2 ou 4)
        image_size: taille du latent (32 pour SD VAE avec images 256x256)
    """
    def __init__(
        self,
        in_channels=4,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        pos_enc="RoPE",
        patch_size=2,
        image_size=32,
    ):
        super().__init__()
        
        assert pos_enc in ["RoPE", "SinPE", "woPE"]
        
        self.pos_enc = pos_enc
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Embedders
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, 
                   pos_enc=pos_enc, max_seq_len=self.num_patches)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, in_channels)
        
        # Positional embedding pour SinPE
        if pos_enc == "SinPE":
            grid_size = image_size // patch_size
            pos_embed = get_2d_sincos_pos_embed(hidden_size, grid_size)
            self.register_buffer("pos_embed", pos_embed, persistent=False)
        
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # PatchEmbed
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedder
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        # Blocks: zero init pour adaLN
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Final layer: zero init
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        Convertit patches → image.
        x: (B, num_patches, patch_size**2 * channels)
        return: (B, channels, H, W)
        """
        c = self.in_channels
        p = self.patch_size
        h = w = self.image_size // p
        
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        return x

    def forward(self, x, t):
        """
        Forward pass.
        x: (B, C, H, W) latent image
        t: (B,) timesteps
        return: (B, C, H, W)
        """
        # Patchify: (B, C, H, W) → (B, num_patches, hidden)
        x = self.x_embedder(x)
        
        # Positional embedding
        if self.pos_enc == "SinPE":
            x = x + self.pos_embed[None, :, :]
        
        # Timestep embedding
        t = self.t_embedder(t)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Final layer + unpatchify
        x = self.final_layer(x, t)
        x = self.unpatchify(x)
        
        return x


#################################################################################
#                             Model Configurations                              #
#################################################################################

def ImageTransformer_S(**kwargs):
    """~33M params"""
    return ImageTransformer(hidden_size=384, depth=12, num_heads=6, **kwargs)

def ImageTransformer_B(**kwargs):
    """~130M params"""
    return ImageTransformer(hidden_size=768, depth=12, num_heads=12, **kwargs)

def ImageTransformer_L(**kwargs):
    """~458M params"""
    return ImageTransformer(hidden_size=1024, depth=24, num_heads=16, **kwargs)

def ImageTransformer_XL(**kwargs):
    """~675M params"""
    return ImageTransformer(hidden_size=1152, depth=28, num_heads=16, **kwargs)