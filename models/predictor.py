"""
JEPA Predictor for Tabular-JEPA.

The predictor is a lightweight transformer that takes context encoder outputs
and mask tokens, then predicts the target encoder representations for the
masked positions.
"""

from typing import Optional

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """
    Lightweight Transformer Predictor for JEPA.
    
    Takes the output of the context encoder (visible features only) along with
    learnable mask tokens at masked positions, and predicts what the target
    encoder would output for those masked positions.
    
    Key design choices:
        - Narrower than encoder (efficiency)
        - Learnable mask tokens carry positional information
        - Separate prediction head projects to target space
    
    Args:
        d_model: Model dimension (same as encoder)
        n_heads: Number of attention heads
        n_layers: Number of transformer layers (typically 1-3)
        d_ff: Feed-forward hidden dimension
        dropout: Dropout rate
        predictor_embed_dim: Internal predictor dimension (None = same as d_model)
    
    Example:
        >>> predictor = JEPAPredictor(d_model=128, n_layers=2)
        >>> context_output = torch.randn(32, 5, 128)  # [batch, num_visible, d_model]
        >>> masked_indices = torch.tensor([[3, 7], ...])  # [batch, num_masked]
        >>> pos_encoding = torch.randn(1, 10, 128)  # [1, num_features, d_model]
        >>> predictions = predictor(context_output, masked_indices, pos_encoding)
        >>> print(predictions.shape)  # [32, 2, 128]
    """
    
    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        predictor_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.predictor_embed_dim = predictor_embed_dim or d_model
        
        # Learnable mask token (one shared token, differentiated by position)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.predictor_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Optional projection if predictor uses different dimension
        if self.predictor_embed_dim != d_model:
            self.input_proj = nn.Linear(d_model, self.predictor_embed_dim)
            self.output_proj = nn.Linear(self.predictor_embed_dim, d_model)
        else:
            self.input_proj = nn.Identity()
            self.output_proj = nn.Identity()
        
        # Predictor transformer layers
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=self.predictor_embed_dim,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=n_layers)
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(self.predictor_embed_dim),
            nn.Linear(self.predictor_embed_dim, d_model),
        )
        
        # Initialize
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        context_output: torch.Tensor,
        masked_indices: torch.Tensor,
        positional_encoding: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict target representations for masked positions.
        
        Args:
            context_output: [batch, num_visible, d_model] - output from context encoder
            masked_indices: [batch, num_masked] - indices of masked features
            positional_encoding: [1, num_features, d_model] - full positional encodings
            visible_indices: Optional [batch, num_visible] - indices of visible features
                            (used to add positional info to context output)
        
        Returns:
            [batch, num_masked, d_model] - predicted target representations
        """
        batch_size = context_output.shape[0]
        num_masked = masked_indices.shape[1]
        device = context_output.device
        
        # Project context output if needed
        context_proj = self.input_proj(context_output)  # [batch, num_visible, pred_dim]
        
        # Create mask tokens with positional information
        # Expand mask token: [batch, num_masked, pred_dim]
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1).clone()
        
        # Add positional encoding for masked positions
        if self.predictor_embed_dim == self.d_model:
            # Gather positional encodings for masked positions
            pos_enc_masked = self._gather_positional_encoding(
                positional_encoding, masked_indices
            )
            mask_tokens = mask_tokens + pos_enc_masked
        
        # Optionally add positional info to context (if not already added by tokenizer)
        if visible_indices is not None and self.predictor_embed_dim == self.d_model:
            pos_enc_visible = self._gather_positional_encoding(
                positional_encoding, visible_indices
            )
            # Note: Usually positional encoding is already in context_output from tokenizer
            # This is just for explicit position injection if needed
        
        # Concatenate: [visible context] + [mask tokens]
        # The predictor attends over all positions to make predictions
        full_sequence = torch.cat([context_proj, mask_tokens], dim=1)
        
        # Run through predictor transformer
        predictor_output = self.predictor(full_sequence)
        
        # Extract only the mask token positions (last num_masked tokens)
        num_visible = context_output.shape[1]
        mask_predictions = predictor_output[:, num_visible:, :]  # [batch, num_masked, pred_dim]
        
        # Final projection to target space
        predictions = self.prediction_head(mask_predictions)  # [batch, num_masked, d_model]
        
        return predictions
    
    def _gather_positional_encoding(
        self,
        positional_encoding: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gather positional encodings for specific indices.
        
        Args:
            positional_encoding: [1, num_features, d_model]
            indices: [batch, num_indices]
            
        Returns:
            [batch, num_indices, d_model]
        """
        batch_size = indices.shape[0]
        d_model = positional_encoding.shape[-1]
        
        # Expand positional encoding for batch
        pe = positional_encoding.expand(batch_size, -1, -1)  # [batch, num_features, d_model]
        
        # Gather
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
        return torch.gather(pe, dim=1, index=indices_expanded)
    
    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, "
            f"n_layers={self.n_layers}, "
            f"predictor_embed_dim={self.predictor_embed_dim}"
        )


class MLPPredictor(nn.Module):
    """
    Simple MLP-based predictor (alternative to transformer predictor).
    
    Faster but may be less expressive for complex relationships.
    
    Args:
        d_model: Model dimension
        hidden_dim: Hidden layer dimension
        n_layers: Number of MLP layers
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # Build MLP layers
        layers = []
        in_dim = d_model
        for i in range(n_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        
        self.mlp = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        context_output: torch.Tensor,
        masked_indices: torch.Tensor,
        positional_encoding: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict target representations using MLP.
        
        This simpler approach:
        1. Pools context output (mean)
        2. Adds positional info for each masked position
        3. Predicts independently per masked position
        
        Args:
            context_output: [batch, num_visible, d_model]
            masked_indices: [batch, num_masked]
            positional_encoding: [1, num_features, d_model]
            visible_indices: Optional, not used in MLP predictor
            
        Returns:
            [batch, num_masked, d_model]
        """
        batch_size = context_output.shape[0]
        num_masked = masked_indices.shape[1]
        device = context_output.device
        
        # Pool context: [batch, d_model]
        context_pooled = context_output.mean(dim=1)
        
        # Create mask tokens with positional info
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1).clone()
        
        # Add positional encoding for masked positions
        pe = positional_encoding.expand(batch_size, -1, -1)
        indices_expanded = masked_indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        pos_enc_masked = torch.gather(pe, dim=1, index=indices_expanded)
        mask_tokens = mask_tokens + pos_enc_masked
        
        # Add pooled context to each mask token
        input_tokens = mask_tokens + context_pooled.unsqueeze(1)
        
        # Apply MLP independently to each position
        # Reshape: [batch * num_masked, d_model]
        input_flat = input_tokens.view(-1, self.d_model)
        output_flat = self.mlp(input_flat)
        
        # Reshape back: [batch, num_masked, d_model]
        predictions = output_flat.view(batch_size, num_masked, self.d_model)
        predictions = self.layer_norm(predictions)
        
        return predictions
