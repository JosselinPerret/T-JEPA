"""
Feature Tokenization for Tabular-JEPA.

Converts raw tabular features (numerical + categorical) into a sequence
of token embeddings suitable for transformer processing.

Architecture:
    - NumericalEmbedding: Each numerical feature → Linear projection to d_model
    - CategoricalEmbedding: Each categorical feature → nn.Embedding lookup
    - FeatureTokenizer: Combines both + adds positional encodings
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    """
    Projects normalized numerical features to d_model dimension.
    
    Each numerical feature gets its own dedicated linear projection,
    allowing the model to learn feature-specific representations.
    
    Args:
        num_numerical: Number of numerical features
        d_model: Embedding dimension
        bias: Whether to include bias in linear projection
    """
    
    def __init__(
        self, 
        num_numerical: int, 
        d_model: int,
        bias: bool = True,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.d_model = d_model
        
        if num_numerical > 0:
            # Each numerical feature gets its own linear projection
            self.projections = nn.ModuleList([
                nn.Linear(1, d_model, bias=bias)
                for _ in range(num_numerical)
            ])
            
            # Initialize with small weights
            self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights."""
        for proj in self.projections:
            nn.init.normal_(proj.weight, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_numerical] - normalized numerical features
            
        Returns:
            [batch, num_numerical, d_model] - embedded features
        """
        if self.num_numerical == 0:
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 0, self.d_model, device=x.device, dtype=x.dtype)
        
        # Project each feature independently
        embeddings = []
        for i, proj in enumerate(self.projections):
            # x[:, i:i+1] keeps dimension: [batch, 1]
            emb = proj(x[:, i:i+1])  # [batch, d_model]
            embeddings.append(emb)
        
        # Stack along feature dimension: [batch, num_numerical, d_model]
        return torch.stack(embeddings, dim=1)


class CategoricalEmbedding(nn.Module):
    """
    Embedding lookup for categorical features.
    
    Each categorical feature has its own embedding table with vocabulary
    size determined by the number of unique categories. Index 0 is reserved
    for unknown/padding.
    
    Args:
        category_sizes: List of vocabulary sizes for each categorical feature
        d_model: Embedding dimension
        padding_idx: Index to use for padding (default: 0)
    """
    
    def __init__(
        self,
        category_sizes: List[int],
        d_model: int,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.category_sizes = category_sizes
        self.num_categorical = len(category_sizes)
        self.d_model = d_model
        
        if self.num_categorical > 0:
            # Separate embedding table per categorical feature
            # +1 for unknown/padding token at index 0
            self.embeddings = nn.ModuleList([
                nn.Embedding(
                    num_embeddings=num_categories + 1,
                    embedding_dim=d_model,
                    padding_idx=padding_idx,
                )
                for num_categories in category_sizes
            ])
            
            # Initialize embeddings
            self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, std=0.02)
            # Zero out padding embedding
            if emb.padding_idx is not None:
                nn.init.zeros_(emb.weight[emb.padding_idx])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_categorical] - integer category indices
            
        Returns:
            [batch, num_categorical, d_model] - embedded features
        """
        if self.num_categorical == 0:
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 0, self.d_model, device=x.device)
        
        # Look up each categorical feature
        embeddings = []
        for i, emb in enumerate(self.embeddings):
            cat_emb = emb(x[:, i])  # [batch, d_model]
            embeddings.append(cat_emb)
        
        # Stack along feature dimension: [batch, num_categorical, d_model]
        return torch.stack(embeddings, dim=1)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encodings for feature positions.
    
    Can use either:
        - Learnable: nn.Parameter (default, recommended for tabular)
        - Sinusoidal: Fixed sin/cos patterns (alternative)
    
    Args:
        num_positions: Maximum number of positions (features)
        d_model: Embedding dimension
        dropout: Dropout rate
        learnable: Whether to use learnable positional encodings
    """
    
    def __init__(
        self,
        num_positions: int,
        d_model: int,
        dropout: float = 0.1,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.d_model = d_model
        self.learnable = learnable
        self.dropout = nn.Dropout(p=dropout)
        
        if learnable:
            # Learnable positional embeddings
            self.positional_encoding = nn.Parameter(
                torch.randn(1, num_positions, d_model) * 0.02
            )
        else:
            # Fixed sinusoidal encodings
            pe = self._create_sinusoidal_encoding(num_positions, d_model)
            self.register_buffer('positional_encoding', pe)
    
    def _create_sinusoidal_encoding(
        self, 
        num_positions: int, 
        d_model: int
    ) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(1, num_positions, d_model)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            
        Returns:
            [batch, seq_len, d_model] with positional encoding added
        """
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:, :seq_len, :]
        return self.dropout(x)
    
    def get_encoding(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get positional encoding for specific indices.
        
        Args:
            indices: [batch, num_indices] or None for all positions
            
        Returns:
            Positional encodings for the specified indices
        """
        if indices is None:
            return self.positional_encoding
        
        # Gather encodings for specific positions
        # indices: [batch, num_indices]
        batch_size = indices.shape[0]
        # Expand positional encoding for batch
        pe = self.positional_encoding.expand(batch_size, -1, -1)  # [batch, num_pos, d_model]
        
        # Gather: [batch, num_indices, d_model]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, self.d_model)
        return torch.gather(pe, dim=1, index=indices_expanded)


class FeatureTokenizer(nn.Module):
    """
    Complete feature tokenization pipeline for tabular data.
    
    Converts raw tabular features into a sequence of token embeddings:
    1. Numerical features → Linear projections
    2. Categorical features → Embedding lookups
    3. Concatenate in feature order
    4. Add positional encodings
    
    Args:
        num_numerical: Number of numerical features
        category_sizes: List of vocab sizes for categorical features
        d_model: Embedding dimension
        feature_order: List of feature names in desired order (for documentation)
        dropout: Dropout rate for positional encoding
        learnable_pos: Whether to use learnable positional encodings
    
    Example:
        >>> tokenizer = FeatureTokenizer(
        ...     num_numerical=5,
        ...     category_sizes=[10, 20, 5],  # 3 categorical features
        ...     d_model=128,
        ... )
        >>> numerical = torch.randn(32, 5)  # [batch, num_numerical]
        >>> categorical = torch.randint(0, 10, (32, 3))  # [batch, num_categorical]
        >>> tokens = tokenizer(numerical, categorical)  # [32, 8, 128]
    """
    
    def __init__(
        self,
        num_numerical: int,
        category_sizes: List[int],
        d_model: int,
        feature_order: Optional[List[str]] = None,
        dropout: float = 0.1,
        learnable_pos: bool = True,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(category_sizes)
        self.num_features = num_numerical + self.num_categorical
        self.d_model = d_model
        self.feature_order = feature_order or [f"feature_{i}" for i in range(self.num_features)]
        
        # Feature embeddings
        self.numerical_embedding = NumericalEmbedding(num_numerical, d_model)
        self.categorical_embedding = CategoricalEmbedding(category_sizes, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            num_positions=self.num_features,
            d_model=d_model,
            dropout=dropout,
            learnable=learnable_pos,
        )
        
        # Layer norm after tokenization (optional but helps stability)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        return_before_pos: bool = False,
    ) -> torch.Tensor:
        """
        Tokenize tabular features.
        
        Args:
            numerical: [batch, num_numerical] - normalized numerical features
            categorical: [batch, num_categorical] - category indices
            return_before_pos: If True, also return embeddings before positional encoding
            
        Returns:
            [batch, num_features, d_model] - tokenized feature sequence
        """
        batch_size = numerical.shape[0]
        
        # Embed numerical features: [batch, num_numerical, d_model]
        num_emb = self.numerical_embedding(numerical)
        
        # Embed categorical features: [batch, num_categorical, d_model]
        cat_emb = self.categorical_embedding(categorical)
        
        # Concatenate: numerical first, then categorical
        # This matches the feature_order from preprocessing
        if self.num_numerical > 0 and self.num_categorical > 0:
            tokens = torch.cat([num_emb, cat_emb], dim=1)
        elif self.num_numerical > 0:
            tokens = num_emb
        else:
            tokens = cat_emb
        
        # Store pre-positional tokens if needed
        tokens_before_pos = tokens if return_before_pos else None
        
        # Add positional encodings
        tokens = self.positional_encoding(tokens)
        
        # Layer norm
        tokens = self.layer_norm(tokens)
        
        if return_before_pos:
            return tokens, tokens_before_pos
        return tokens
    
    def get_positional_encoding(
        self, 
        indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get positional encodings for specific feature indices."""
        return self.positional_encoding.get_encoding(indices)
    
    @classmethod
    def from_config(cls, config: dict, **kwargs) -> "FeatureTokenizer":
        """
        Create tokenizer from preprocessor config.
        
        Args:
            config: Dictionary from TabularPreprocessor.get_config()
            **kwargs: Additional arguments (d_model, dropout, etc.)
            
        Returns:
            Configured FeatureTokenizer
        """
        return cls(
            num_numerical=config['num_numerical'],
            category_sizes=config['category_sizes'],
            feature_order=config.get('feature_order'),
            **kwargs,
        )
    
    def extra_repr(self) -> str:
        return (
            f"num_numerical={self.num_numerical}, "
            f"num_categorical={self.num_categorical}, "
            f"d_model={self.d_model}"
        )
