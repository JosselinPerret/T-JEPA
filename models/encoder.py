"""
Transformer Encoder for Tabular-JEPA.

Implements a ViT-style transformer encoder that processes the tokenized
tabular features. Used as both Context Encoder and Target Encoder.
"""

from typing import Optional

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block with pre-norm architecture.
    
    Structure:
        x → LayerNorm → MultiHeadAttention → Dropout → + → LayerNorm → FFN → Dropout → +
        ↑_______________________________________________↑__________________________________↑
    
    Args:
        d_model: Embedding dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        activation: Activation function (gelu, relu)
        pre_norm: Whether to use pre-norm (True) or post-norm (False)
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for residual
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            attn_mask: Optional attention mask
            key_padding_mask: Optional padding mask [batch, seq_len]
            
        Returns:
            [batch, seq_len, d_model]
        """
        if self.pre_norm:
            # Pre-norm: Norm before attention/FFN
            # Self-attention block
            x_norm = self.norm1(x)
            attn_out, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = x + self.dropout(attn_out)
            
            # FFN block
            x_norm = self.norm2(x)
            x = x + self.ffn(x_norm)
        else:
            # Post-norm: Norm after attention/FFN
            attn_out, _ = self.self_attn(
                x, x, x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.ffn(x))
        
        return x


class TransformerEncoder(nn.Module):
    """
    ViT-style Transformer Encoder for tabular tokens.
    
    Processes a sequence of feature tokens and outputs contextualized
    representations for each position.
    
    Args:
        d_model: Embedding dimension (default: 128)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer layers (default: 6)
        d_ff: Feed-forward hidden dimension (default: 512)
        dropout: Dropout rate (default: 0.1)
        activation: Activation function (default: "gelu")
        pre_norm: Whether to use pre-norm architecture (default: True)
    
    Example:
        >>> encoder = TransformerEncoder(d_model=128, n_layers=6)
        >>> x = torch.randn(32, 10, 128)  # [batch, num_features, d_model]
        >>> output = encoder(x)  # [32, 10, 128]
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 6,
        d_ff: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.pre_norm = pre_norm
        
        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                pre_norm=pre_norm,
            )
            for _ in range(n_layers)
        ])
        
        # Final layer norm (for pre-norm architecture)
        if pre_norm:
            self.final_norm = nn.LayerNorm(d_model)
        else:
            self.final_norm = nn.Identity()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize transformer weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Args:
            x: [batch, seq_len, d_model] - input token embeddings
            mask: Optional attention mask
            return_all_layers: If True, return outputs from all layers
            
        Returns:
            [batch, seq_len, d_model] - contextualized representations
            Or list of layer outputs if return_all_layers=True
        """
        all_hidden_states = [] if return_all_layers else None
        
        for layer in self.layers:
            x = layer(x, attn_mask=mask)
            if return_all_layers:
                all_hidden_states.append(x)
        
        x = self.final_norm(x)
        
        if return_all_layers:
            all_hidden_states[-1] = x  # Replace last with normed version
            return all_hidden_states
        
        return x
    
    def get_attention_maps(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Get attention maps from a specific layer (for visualization).
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            layer_idx: Which layer to get attention from (-1 = last)
            
        Returns:
            [batch, n_heads, seq_len, seq_len] attention weights
        """
        # Run through layers up to target
        for i, layer in enumerate(self.layers):
            if i == layer_idx or (layer_idx == -1 and i == len(self.layers) - 1):
                # Get attention from this layer
                x_norm = layer.norm1(x) if layer.pre_norm else x
                _, attn_weights = layer.self_attn(
                    x_norm, x_norm, x_norm,
                    need_weights=True,
                    average_attn_weights=False,
                )
                return attn_weights
            x = layer(x)
        
        return None
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_layers={self.n_layers}"
