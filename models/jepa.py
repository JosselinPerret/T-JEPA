"""
Complete Tabular-JEPA Model.

Assembles all components into the full JEPA architecture:
- Feature Tokenizer (shared)
- Context Encoder (learnable via backprop)
- Target Encoder (EMA of context encoder, no gradients)
- Predictor (learnable)
"""

import copy
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn

from .tokenizer import FeatureTokenizer
from .encoder import TransformerEncoder
from .predictor import JEPAPredictor, MLPPredictor
from .masking import RandomMasking, MaskingOutput, create_masking_strategy


class TabularJEPA(nn.Module):
    """
    Complete Tabular-JEPA model for self-supervised pre-training.
    
    Architecture:
        Input → Tokenizer → [visible_tokens, masked_tokens]
                               ↓                ↓
                        Context Encoder   Target Encoder (EMA)
                               ↓                ↓
                           Predictor  ←→    Targets
                               ↓                ↓
                            L2 Loss in Latent Space
    
    Args:
        num_numerical: Number of numerical features
        category_sizes: List of vocab sizes for categorical features
        d_model: Embedding dimension (default: 128)
        encoder_layers: Number of transformer layers in encoder (default: 6)
        encoder_heads: Number of attention heads in encoder (default: 4)
        encoder_ff_dim: Feed-forward dimension in encoder (default: 512)
        predictor_layers: Number of transformer layers in predictor (default: 2)
        predictor_heads: Number of attention heads in predictor (default: 4)
        predictor_ff_dim: Feed-forward dimension in predictor (default: 256)
        dropout: Dropout rate (default: 0.1)
        mask_ratio: Fraction of features to mask (default: 0.5)
        mask_type: Type of masking ('random', 'block') (default: 'random')
        ema_decay: EMA decay for target encoder (default: 0.996)
        predictor_type: Type of predictor ('transformer', 'mlp') (default: 'transformer')
        feature_order: List of feature names (optional, for documentation)
    
    Example:
        >>> model = TabularJEPA(
        ...     num_numerical=5,
        ...     category_sizes=[10, 20, 5],
        ...     d_model=128,
        ... )
        >>> numerical = torch.randn(32, 5)
        >>> categorical = torch.randint(1, 5, (32, 3))
        >>> output = model(numerical, categorical)
        >>> print(output['predictions'].shape)  # [32, 4, 128] (4 masked features)
        >>> print(output['targets'].shape)      # [32, 4, 128]
    """
    
    def __init__(
        self,
        num_numerical: int,
        category_sizes: list,
        d_model: int = 128,
        encoder_layers: int = 6,
        encoder_heads: int = 4,
        encoder_ff_dim: int = 512,
        predictor_layers: int = 2,
        predictor_heads: int = 4,
        predictor_ff_dim: int = 256,
        dropout: float = 0.1,
        mask_ratio: float = 0.5,
        mask_type: str = "random",
        ema_decay: float = 0.996,
        predictor_type: str = "transformer",
        feature_order: Optional[list] = None,
    ):
        super().__init__()
        
        self.num_numerical = num_numerical
        self.num_categorical = len(category_sizes)
        self.num_features = num_numerical + len(category_sizes)
        self.d_model = d_model
        self.ema_decay = ema_decay
        
        # ===== Feature Tokenizer (shared) =====
        self.tokenizer = FeatureTokenizer(
            num_numerical=num_numerical,
            category_sizes=category_sizes,
            d_model=d_model,
            feature_order=feature_order,
            dropout=dropout,
        )
        
        # ===== Context Encoder (learnable via backprop) =====
        self.context_encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=encoder_heads,
            n_layers=encoder_layers,
            d_ff=encoder_ff_dim,
            dropout=dropout,
        )
        
        # ===== Target Encoder (EMA of context encoder) =====
        # Deep copy and freeze
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self._freeze_target_encoder()
        
        # ===== Predictor =====
        if predictor_type == "transformer":
            self.predictor = JEPAPredictor(
                d_model=d_model,
                n_heads=predictor_heads,
                n_layers=predictor_layers,
                d_ff=predictor_ff_dim,
                dropout=dropout,
            )
        elif predictor_type == "mlp":
            self.predictor = MLPPredictor(
                d_model=d_model,
                hidden_dim=predictor_ff_dim,
                n_layers=predictor_layers,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        # ===== Masking Strategy =====
        if mask_type == "structured":
            self.masking = create_masking_strategy(
                mask_type,
                mask_ratio=mask_ratio,
                num_numerical=num_numerical,
                num_categorical=len(category_sizes),
            )
        else:
            self.masking = create_masking_strategy(mask_type, mask_ratio=mask_ratio)
    
    def _freeze_target_encoder(self):
        """Freeze target encoder parameters (no gradients)."""
        for param in self.target_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        """
        Update target encoder via Exponential Moving Average.
        
        θ_target = decay * θ_target + (1 - decay) * θ_context
        
        Args:
            decay: EMA decay value (uses self.ema_decay if not specified)
        """
        decay = decay if decay is not None else self.ema_decay
        
        for param_context, param_target in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_target.data.mul_(decay).add_(param_context.data, alpha=1 - decay)
    
    def forward(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        mask_output: Optional[MaskingOutput] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for self-supervised pre-training.
        
        Args:
            numerical: [batch, num_numerical] - normalized numerical features
            categorical: [batch, num_categorical] - category indices
            mask_output: Optional pre-computed masking (for consistency in training)
        
        Returns:
            Dictionary containing:
                - 'predictions': [batch, num_masked, d_model] - predictor output
                - 'targets': [batch, num_masked, d_model] - target encoder output (detached)
                - 'masked_indices': [batch, num_masked] - indices of masked features
                - 'visible_indices': [batch, num_visible] - indices of visible features
        """
        # Step 1: Tokenize all features
        tokens = self.tokenizer(numerical, categorical)  # [batch, num_features, d_model]
        
        # Step 2: Apply masking
        if mask_output is None:
            mask_output = self.masking(tokens)
        
        # Step 3: Context Encoder - process visible tokens only
        context_output = self.context_encoder(mask_output.visible_tokens)
        # [batch, num_visible, d_model]
        
        # Step 4: Target Encoder - process masked tokens (no grad)
        with torch.no_grad():
            target_output = self.target_encoder(mask_output.masked_tokens)
            # [batch, num_masked, d_model]
        
        # Step 5: Predictor - predict target representations
        positional_encoding = self.tokenizer.positional_encoding.positional_encoding
        predictions = self.predictor(
            context_output=context_output,
            masked_indices=mask_output.masked_indices,
            positional_encoding=positional_encoding,
            visible_indices=mask_output.visible_indices,
        )
        # [batch, num_masked, d_model]
        
        return {
            'predictions': predictions,
            'targets': target_output.detach(),  # Ensure no gradients to targets
            'masked_indices': mask_output.masked_indices,
            'visible_indices': mask_output.visible_indices,
            'context_output': context_output,  # Useful for analysis
        }
    
    def get_representations(
        self,
        numerical: torch.Tensor,
        categorical: torch.Tensor,
        pooling: str = "mean",
        use_context_encoder: bool = True,
    ) -> torch.Tensor:
        """
        Extract representations for downstream tasks.
        
        Processes ALL features (no masking) through the encoder.
        
        Args:
            numerical: [batch, num_numerical]
            categorical: [batch, num_categorical]
            pooling: How to pool features ('mean', 'first', 'cls', 'none')
            use_context_encoder: Use context (True) or target (False) encoder
        
        Returns:
            [batch, d_model] if pooling != 'none'
            [batch, num_features, d_model] if pooling == 'none'
        """
        # Tokenize
        tokens = self.tokenizer(numerical, categorical)
        
        # Encode all features
        encoder = self.context_encoder if use_context_encoder else self.target_encoder
        
        with torch.no_grad() if not use_context_encoder else torch.enable_grad():
            representations = encoder(tokens)  # [batch, num_features, d_model]
        
        # Pool
        if pooling == "none":
            return representations
        elif pooling == "mean":
            return representations.mean(dim=1)  # [batch, d_model]
        elif pooling == "first":
            return representations[:, 0, :]  # [batch, d_model]
        elif pooling == "cls":
            # Same as first for tabular (no CLS token by default)
            return representations[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
    
    def freeze_encoder(self) -> None:
        """Freeze tokenizer and context encoder for linear probing."""
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        for param in self.context_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze tokenizer and context encoder for fine-tuning."""
        for param in self.tokenizer.parameters():
            param.requires_grad = True
        for param in self.context_encoder.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TabularJEPA":
        """
        Create model from configuration dictionary.
        
        Args:
            config: Configuration with keys matching __init__ parameters
            
        Returns:
            Configured TabularJEPA model
        """
        return cls(**config)
    
    @classmethod
    def from_pretrained(cls, path: str, map_location: str = "cpu") -> "TabularJEPA":
        """
        Load pre-trained model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            map_location: Device to load to
            
        Returns:
            Loaded TabularJEPA model
        """
        checkpoint = torch.load(path, map_location=map_location)
        
        # Reconstruct model from saved config
        model = cls.from_config(checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def save_pretrained(self, path: str) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'model_config': {
                'num_numerical': self.num_numerical,
                'category_sizes': self.tokenizer.categorical_embedding.category_sizes,
                'd_model': self.d_model,
                'encoder_layers': self.context_encoder.n_layers,
                'encoder_heads': self.context_encoder.layers[0].self_attn.num_heads,
                'predictor_layers': self.predictor.n_layers if hasattr(self.predictor, 'n_layers') else 2,
                'ema_decay': self.ema_decay,
            },
            'model_state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
    
    def get_num_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"d_model={self.d_model}, "
            f"ema_decay={self.ema_decay}"
        )
