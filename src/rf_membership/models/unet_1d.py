"""
U-Net 1D pour Rectified Flow sur latents audio (Stable Audio VAE).

Adapté de UNet2D - même structure, même interface.
Pour latents (64, T) de Stable Audio VAE.

Interface: forward(x, t) → (B, C, T)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#################################################################################
#               Timestep Embedding                                              #
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
        t_freq = t_freq.to(next(self.mlp.parameters()).dtype)
        return self.mlp(t_freq)


#################################################################################
#                              Helper Functions                                 #
#################################################################################

def get_num_groups(channels, target_groups=32):
    """Trouve un nombre de groupes valide pour GroupNorm."""
    while channels % target_groups != 0 and target_groups > 1:
        target_groups //= 2
    return max(target_groups, 1)


#################################################################################
#                              ResNet Block 1D                                  #
#################################################################################

class ResBlock1D(nn.Module):
    """ResBlock 1D avec time conditioning (scale & shift)."""
    
    def __init__(self, in_ch, out_ch, time_emb_dim, dropout=0.0):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.norm1 = nn.GroupNorm(get_num_groups(in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(get_num_groups(out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch * 2)
        )
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if in_ch != out_ch:
            self.skip_conv = nn.Conv1d(in_ch, out_ch, 1)
        else:
            self.skip_conv = nn.Identity()
        
        # Zero init dernière conv
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, x, t_emb):
        """
        Args:
            x: (B, C, T)
            t_emb: (B, time_emb_dim)
        """
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time conditioning: (B, out_ch * 2) → (B, out_ch * 2, 1)
        t_out = self.time_mlp(t_emb)[:, :, None]
        scale, shift = t_out.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.skip_conv(x)


#################################################################################
#                            Attention Block 1D                                 #
#################################################################################

class AttentionBlock1D(nn.Module):
    """Self-attention pour séquences 1D."""
    
    def __init__(self, channels, num_heads=8):
        super().__init__()
        
        # Adapter num_heads si nécessaire
        while channels % num_heads != 0 and num_heads > 1:
            num_heads //= 2
        self.num_heads = max(num_heads, 1)
        
        self.norm = nn.GroupNorm(get_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, self.num_heads, batch_first=True)
        
        # Zero init
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T)
        """
        B, C, T = x.shape
        
        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, T, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2)  # (B, C, T)
        
        return x + h


#################################################################################
#                          Downsample / Upsample 1D                             #
#################################################################################

class Downsample1D(nn.Module):
    """Downsample par 2 sur la dimension temporelle."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample par 2 sur la dimension temporelle."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


#################################################################################
#                                  U-Net 1D                                     #
#################################################################################

class UNet1D(nn.Module):
    """
    U-Net 1D pour Rectified Flow sur latents audio.
    
    Adapté de UNet2D avec la même structure.
    
    Args:
        in_channels: canaux d'entrée (64 pour Stable Audio VAE)
        model_channels: canaux de base
        channel_mult: multiplicateurs par niveau (ex: (1, 2, 4, 4))
        num_res_blocks: ResBlocks par niveau
        attention_levels: indices des niveaux avec attention (ex: (2, 3))
        dropout: dropout rate
        num_heads: têtes d'attention
    """
    
    def __init__(
        self,
        in_channels=64,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        num_res_blocks=2,
        attention_levels=(2, 3),
        dropout=0.0,
        num_heads=8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.num_levels = len(channel_mult)
        self.attention_levels = set(attention_levels)
        
        time_emb_dim = model_channels * 4
        
        # Time embedding
        self.time_embed = TimestepEmbedder(time_emb_dim)
        
        # Input
        self.input_conv = nn.Conv1d(in_channels, model_channels, 3, padding=1)
        
        # ================== ENCODER ==================
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        # Track channels pour les skip connections
        skip_channels = [model_channels]  # après input_conv
        
        ch = model_channels
        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]
            
            # ResBlocks pour ce niveau
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock1D(ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                if level in self.attention_levels:
                    level_attns.append(AttentionBlock1D(ch, num_heads))
                else:
                    level_attns.append(nn.Identity())
                
                skip_channels.append(ch)
            
            self.encoder_blocks.append(level_blocks)
            self.encoder_attns.append(level_attns)
            
            # Downsample (sauf dernier niveau)
            if level < self.num_levels - 1:
                self.encoder_downsamples.append(Downsample1D(ch))
                skip_channels.append(ch)
            else:
                self.encoder_downsamples.append(nn.Identity())
        
        # ================== MIDDLE ==================
        self.mid_block1 = ResBlock1D(ch, ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock1D(ch, num_heads)
        self.mid_block2 = ResBlock1D(ch, ch, time_emb_dim, dropout)
        
        # ================== DECODER ==================
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]
            
            level_blocks = nn.ModuleList()
            level_attns = nn.ModuleList()
            
            # num_res_blocks + 1 car on a un skip supplémentaire après downsample
            for i in range(num_res_blocks + 1):
                skip_ch = skip_channels.pop()
                level_blocks.append(ResBlock1D(ch + skip_ch, out_ch, time_emb_dim, dropout))
                ch = out_ch
                
                if level in self.attention_levels:
                    level_attns.append(AttentionBlock1D(ch, num_heads))
                else:
                    level_attns.append(nn.Identity())
            
            self.decoder_blocks.append(level_blocks)
            self.decoder_attns.append(level_attns)
            
            # Upsample (sauf dernier niveau = level 0)
            if level > 0:
                self.decoder_upsamples.append(Upsample1D(ch))
            else:
                self.decoder_upsamples.append(nn.Identity())
        
        # Vérifier que tous les skip_channels ont été utilisés
        assert len(skip_channels) == 0, f"Skip channels mismatch: {len(skip_channels)} remaining"
        
        # ================== OUTPUT ==================
        self.out_norm = nn.GroupNorm(get_num_groups(ch), ch)
        self.out_conv = nn.Conv1d(ch, in_channels, 3, padding=1)
        
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t):
        """
        Args:
            x: (B, C, T) - noisy latents
            t: (B,) - timesteps [0, 1]
        Returns:
            (B, C, T) - predicted velocity
        """
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Input
        h = self.input_conv(x)
        
        # ================== ENCODER ==================
        skips = [h]
        
        for level in range(self.num_levels):
            # ResBlocks
            for block, attn in zip(self.encoder_blocks[level], self.encoder_attns[level]):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            
            # Downsample
            if level < self.num_levels - 1:
                h = self.encoder_downsamples[level](h)
                skips.append(h)
        
        # ================== MIDDLE ==================
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # ================== DECODER ==================
        for level_idx, level in enumerate(reversed(range(self.num_levels))):
            # ResBlocks avec skip connections
            for block, attn in zip(self.decoder_blocks[level_idx], self.decoder_attns[level_idx]):
                skip = skips.pop()
                # Gérer les différences de taille dues au downsampling/upsampling
                if h.shape[-1] != skip.shape[-1]:
                    diff = skip.shape[-1] - h.shape[-1]
                    if diff > 0:
                        h = F.pad(h, (0, diff))
                    else:
                        h = h[..., :skip.shape[-1]]
                h = torch.cat([h, skip], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            
            # Upsample
            if level > 0:
                h = self.decoder_upsamples[level_idx](h)
        
        # ================== OUTPUT ==================
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h



