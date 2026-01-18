"""
DiT (Diffusion Transformer) style blocks with Adaptive Layer Normalization (AdaLN).

Reference: "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023)

Key insight: Instead of concatenating condition tokens with input tokens,
AdaLN injects conditions at EVERY layer by modulating the LayerNorm parameters.
This provides stronger and more direct conditioning throughout the network.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


def modulate(x, shift, scale):
    """Apply adaptive modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """Simple MLP with GELU activation."""

    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with Adaptive Layer Normalization (AdaLN-Zero).

    The key difference from standard Transformer:
    - LayerNorm has no learnable parameters (elementwise_affine=False)
    - Scale and shift are generated from the condition vector
    - Gate mechanism controls residual contribution

    Args:
        hidden_size: Dimension of the model
        num_heads: Number of attention heads
        cond_dim: Dimension of the condition vector
        mlp_ratio: Ratio of MLP hidden dim to hidden_size
        drop: Dropout rate
        drop_path: Stochastic depth rate
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # LayerNorm without learnable parameters
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Self-Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            drop=drop
        )

        # DropPath for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # AdaLN modulation: generates 6 parameters (shift, scale, gate) x 2 branches
        # Initialized to zero so that initial behavior is like standard residual
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size, bias=True)
        )
        # Zero-initialize the last linear layer
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond, padding_mask=None):
        """
        Args:
            x: Input tensor [B, L, D]
            cond: Condition vector [B, cond_dim]
            padding_mask: Optional mask [B, L], True for positions to mask

        Returns:
            Output tensor [B, L, D]
        """
        # Generate modulation parameters from condition
        # [B, 6*D] -> 6 x [B, D]
        modulation = self.adaLN_modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            modulation.chunk(6, dim=-1)

        # Self-Attention branch with AdaLN
        h = self.norm1(x)
        h = modulate(h, shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, key_padding_mask=padding_mask)
        x = x + gate_msa.unsqueeze(1) * self.drop_path(h)

        # MLP branch with AdaLN
        h = self.norm2(x)
        h = modulate(h, shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp.unsqueeze(1) * self.drop_path(h)

        return x


class DiTBlockWithCrossAttention(nn.Module):
    """
    DiT block with both self-attention and cross-attention.

    This variant adds cross-attention to external context (e.g., contact features),
    providing an additional pathway for condition injection beyond AdaLN.

    Architecture:
        1. AdaLN + Self-Attention
        2. Cross-Attention to context
        3. AdaLN + MLP
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        cond_dim: int,
        context_dim: int = None,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        context_dim = context_dim or hidden_size

        # Norms
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(context_dim, eps=1e-6)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=drop,
            batch_first=True
        )

        # Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            kdim=context_dim,
            vdim=context_dim,
            dropout=drop,
            batch_first=True
        )

        # MLP
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = MLP(hidden_size, mlp_hidden_dim, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # AdaLN: 8 parameters (self-attn, cross-attn scale/shift, MLP scale/shift/gate x2)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 8 * hidden_size, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond, context=None, padding_mask=None, context_mask=None):
        """
        Args:
            x: Input tensor [B, L, D]
            cond: Condition vector [B, cond_dim]
            context: Context for cross-attention [B, N, context_dim]
            padding_mask: Mask for x [B, L]
            context_mask: Mask for context [B, N]
        """
        # Generate modulation parameters
        modulation = self.adaLN_modulation(cond)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp = \
            modulation.chunk(8, dim=-1)
        gate_mlp = torch.ones_like(scale_mlp)  # Simplification: no gate for MLP

        # Self-Attention
        h = self.norm1(x)
        h = modulate(h, shift_sa, scale_sa)
        h, _ = self.self_attn(h, h, h, key_padding_mask=padding_mask)
        x = x + gate_sa.unsqueeze(1) * self.drop_path(h)

        # Cross-Attention (if context provided)
        if context is not None:
            h = self.norm2(x)
            h = modulate(h, shift_ca, scale_ca)
            context_normed = self.norm_context(context)
            h, _ = self.cross_attn(h, context_normed, context_normed, key_padding_mask=context_mask)
            x = x + gate_ca.unsqueeze(1) * self.drop_path(h)

        # MLP
        h = self.norm3(x)
        h = modulate(h, shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + self.drop_path(h)

        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT with AdaLN.
    """

    def __init__(self, hidden_size: int, output_size: int, cond_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        )
        # Zero-initialize
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ConditionEmbedder(nn.Module):
    """
    Fuses multiple condition embeddings into a single condition vector.

    Takes: time_emb [B, D], text_emb [B, 1, D], contact_emb [B, N, D]
    Returns: cond [B, cond_dim]
    """

    def __init__(self, input_dim: int, output_dim: int, use_cross_attn_pooling: bool = True):
        super().__init__()
        self.use_cross_attn_pooling = use_cross_attn_pooling

        if use_cross_attn_pooling:
            # Learnable query to aggregate contact features
            self.contact_query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)
            self.contact_attn = nn.MultiheadAttention(
                embed_dim=input_dim,
                num_heads=8,
                batch_first=True
            )
            self.contact_norm = nn.LayerNorm(input_dim)

        # Final projection: time + text + contact -> cond_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 3, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, time_emb, text_emb, contact_emb, contact_mask=None):
        """
        Args:
            time_emb: [B, D]
            text_emb: [B, 1, D] or [B, D]
            contact_emb: [B, N, D]
            contact_mask: [B, N], True for masked positions

        Returns:
            cond: [B, cond_dim]
        """
        B = time_emb.shape[0]

        # Time embedding
        if time_emb.dim() == 3:
            time_emb = time_emb.squeeze(1)  # [B, D]

        # Text embedding (take [CLS] or first token)
        if text_emb.dim() == 3:
            text_emb = text_emb[:, 0, :]  # [B, D]

        # Contact embedding aggregation
        if self.use_cross_attn_pooling:
            # Use cross-attention with learnable query
            query = self.contact_query.expand(B, -1, -1)  # [B, 1, D]
            contact_pooled, _ = self.contact_attn(
                query, contact_emb, contact_emb,
                key_padding_mask=contact_mask
            )
            contact_pooled = self.contact_norm(contact_pooled).squeeze(1)  # [B, D]
        else:
            # Simple mean pooling
            if contact_mask is not None:
                mask_expanded = (~contact_mask).unsqueeze(-1).float()
                contact_pooled = (contact_emb * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1)
            else:
                contact_pooled = contact_emb.mean(dim=1)  # [B, D]

        # Concatenate and project
        cond = torch.cat([time_emb, text_emb, contact_pooled], dim=-1)  # [B, 3*D]
        cond = self.proj(cond)  # [B, cond_dim]

        return cond


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Uses sinusoidal positional embedding followed by MLP.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: 1D Tensor of N indices, one per batch element.
            dim: The dimension of the output.
            max_period: Controls the minimum frequency.

        Returns:
            [N, dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
