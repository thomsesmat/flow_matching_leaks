# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        # utiliser le même dtype que t, pas float32 forcé
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=t.dtype, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # aligner le dtype sur la MLP au cas où
        t_freq = t_freq.to(next(self.mlp.parameters()).dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LinearProjection(nn.Module):
    def __init__(self, in_dim, hidden_size):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_size)

    def forward(self, x):
        # x: (batch, in_dim)
        return self.proj(x)  # (batch, hidden_size)
    

# -----------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# Implementation with CLAUDE AI
# -----------------------------------------------------------
def build_rotary_frequencies(dim, max_seq_len=8192, base=10000, device=None):
    """Construit les fréquences pour RoPE (version sin/cos compatible DDP)."""
    assert dim % 2 == 0, "La dimension doit être paire pour RoPE"
    
    half_dim = dim // 2
    freq_inv = 1.0 / (base ** (torch.arange(0, half_dim, dtype=torch.float32, device=device) / half_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freq_inv)
    
    # Au lieu de torch.polar, on retourne cos et sin séparément
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    
    return freqs_cos, freqs_sin


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Applique RoPE à un tenseur (version sin/cos)."""
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    
    seq_len = x_reshaped.shape[-3]  # Obtenir la dimension seq_len
    
    # Tronquer les fréquences à la longueur de séquence actuelle
    freqs_cos = freqs_cos[:seq_len, :]
    freqs_sin = freqs_sin[:seq_len, :]
    
    # Extraire les parties "réelle" et "imaginaire"
    x_real = x_reshaped[..., 0]  # [..., seq_len, dim//2]
    x_imag = x_reshaped[..., 1]  # [..., seq_len, dim//2]
    
    # Appliquer la rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i(a*sin + b*cos)
    x_rotated_real = x_real * freqs_cos - x_imag * freqs_sin
    x_rotated_imag = x_real * freqs_sin + x_imag * freqs_cos
    
    # Recombiner en empilant réel et imaginaire
    x_out = torch.stack([x_rotated_real, x_rotated_imag], dim=-1)
    # Aplatir les deux dernières dimensions
    x_out = x_out.flatten(-2)  # [..., seq_len, dim]
    
    return x_out.type_as(x)


# -----------------------------------------------------------
# Flash Attention avec RoPE
# Implementation with CLAUDE AI
# -----------------------------------------------------------
class FlashAttentionWithRoPE(nn.Module):
    """
    Attention multi-têtes avec Flash Attention ET RoPE.
    Compatible avec adaLN-Zero conditioning.
    """
    
    def __init__(
        self, 
        dim, 
        num_heads=8, 
        qkv_bias=True,
        dropout=0.0,
        max_seq_len=8192,
        rope_base=10000
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim doit être divisible par num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        
        # Projections Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Précalculer les fréquences RoPE
        freqs_cos, freqs_sin = build_rotary_frequencies(
            self.head_dim, 
            max_seq_len=max_seq_len,
            base=rope_base
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    
    def forward(self, x, is_causal=False):
        """
        Args:
            x: tenseur de forme (B, T, C)
            is_causal: si True, applique un masque causal
        
        Returns:
            tenseur de forme (B, T, C)
        """
        B, T, C = x.shape
        
        # Projection QKV
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Appliquer RoPE sur Q et K
        q = apply_rotary_emb(q, self.freqs_cos, self.freqs_sin)
        k = apply_rotary_emb(k, self.freqs_cos, self.freqs_sin)
        
        # Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Reshape et projection
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        
        return out





#################################################################################
#                                 Core rft Model                                #
# Based initialy on DiT from Meta                                               #
#################################################################################

class TBlock(nn.Module):
    """
    A Transformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, pos_enc= "RoPE", **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # ROPE attention
        if pos_enc == "RoPE":
            self.attn = FlashAttentionWithRoPE(dim=hidden_size, num_heads=num_heads, qkv_bias=True, 
                                            dropout=0.0, max_seq_len=215, rope_base=10000,**block_kwargs)
            
        elif pos_enc == "SinPE" or pos_enc == "woPE":
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


class FinalLayer(nn.Module):
    """
    The final layer of rft.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size, bias=True)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)  
        return x



class Transformer(nn.Module):
    """
    Transformer backbone for models
    """
    def __init__(
        self,
        input_size=64,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        pos_enc= "RoPE" # RoPE, SinPE or woPE (without PE)
    ):
        super().__init__()

        assert pos_enc in ["RoPE", "SinPE", "woPE"], "pos_enc should wheter RoPE, SinPE or woPe (i.e. wihtout any PE)"
        self.pos_enc = pos_enc

        self.num_heads = num_heads


        self.x_embedder = LinearProjection(input_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.output_latent = LinearProjection(hidden_size, input_size)

        self.blocks = nn.ModuleList([
            TBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, pos_enc=self.pos_enc) for _ in range(depth)
        ])

        if self.pos_enc == "SinPE":
                    # positional encoding sin/cos précomputé, non appris
            self.register_buffer(
                "pos_embed",
                get_sincos_embedding(50, hidden_size),  # (seq_len, hidden_size)
                persistent=False,
            )


        self.final_layer = FinalLayer(hidden_size)

        self.initialize_weights()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.x_embedder.proj.weight, std=0.02)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.normal_(self.output_latent.proj.weight, std=0.02)
        nn.init.constant_(self.output_latent.proj.bias, 0)
        
        nn.init.constant_(self.final_layer.adaLN_modulation.weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation.bias, 0)


    def forward(self, x, t):
        """
        Forward pass of rft.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if len(x.shape) == 4: 
            x = x.squeeze(1)
        x = x.permute(0, 2, 1)            # (Batch, 64, 50) --> (Batch, 50, 64)
        x = self.x_embedder(x)  

        if self.pos_enc == "SinPE":
            x = x + self.pos_embed[None, :, :]    # ajout position absolue

        t = self.t_embedder(t)              

        for block in self.blocks:
            x = block(x, t)                      # (N, T, D)
    
        x = self.final_layer(x, t)               # modulation finale adaLN
        x = self.output_latent(x) 

        x = x.permute(0, 2, 1)            # (Batch, 50, 64) --> (Batch, 64, 50)
    
        return x


def get_sincos_embedding(n_pos, dim, max_period=10000):
    position = torch.arange(n_pos).float().unsqueeze(1)                   # (n_pos, 1)
    div_term = torch.exp(
            -math.log(max_period) * torch.arange(0, dim//2).float() / (dim//2)
            )                                                                      # (dim/2,)
    pe = torch.zeros(n_pos, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe