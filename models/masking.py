"""
Masking strategies for Tabular-JEPA.

Provides different masking approaches for self-supervised pre-training:
- RandomMasking: Randomly select features to mask
- BlockMasking: Mask contiguous blocks of features
- StructuredMasking: Mask based on feature groups (e.g., all numerical)
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import random

import torch


@dataclass
class MaskingOutput:
    """
    Container for masking results.
    
    Attributes:
        visible_indices: [batch, num_visible] - indices of visible features
        masked_indices: [batch, num_masked] - indices of masked features
        visible_tokens: [batch, num_visible, d_model] - visible token embeddings
        masked_tokens: [batch, num_masked, d_model] - masked token embeddings (targets)
        visible_mask: [batch, num_features] - boolean mask (True = visible)
    """
    visible_indices: torch.Tensor
    masked_indices: torch.Tensor
    visible_tokens: torch.Tensor
    masked_tokens: torch.Tensor
    visible_mask: torch.Tensor


class RandomMasking:
    """
    Random feature masking for JEPA.
    
    Randomly selects a subset of features to mask. The visible features
    are processed by the context encoder, while masked feature representations
    from the target encoder serve as prediction targets.
    
    Args:
        mask_ratio: Fraction of features to mask (default: 0.5)
        min_masked: Minimum number of features to mask (default: 1)
        min_visible: Minimum number of features to keep visible (default: 1)
    
    Example:
        >>> masking = RandomMasking(mask_ratio=0.5)
        >>> tokens = torch.randn(32, 10, 128)  # [batch, num_features, d_model]
        >>> output = masking(tokens)
        >>> print(output.visible_tokens.shape)  # [32, 5, 128]
        >>> print(output.masked_tokens.shape)   # [32, 5, 128]
    """
    
    def __init__(
        self,
        mask_ratio: float = 0.5,
        min_masked: int = 1,
        min_visible: int = 1,
    ):
        assert 0 < mask_ratio < 1, "mask_ratio must be between 0 and 1"
        self.mask_ratio = mask_ratio
        self.min_masked = min_masked
        self.min_visible = min_visible
    
    def __call__(self, tokens: torch.Tensor) -> MaskingOutput:
        """
        Apply random masking to tokens.
        
        Args:
            tokens: [batch, num_features, d_model] - tokenized features
            
        Returns:
            MaskingOutput with visible/masked splits
        """
        batch_size, num_features, d_model = tokens.shape
        device = tokens.device
        
        # Calculate number to mask
        num_masked = max(
            self.min_masked,
            int(num_features * self.mask_ratio)
        )
        num_masked = min(num_masked, num_features - self.min_visible)
        num_visible = num_features - num_masked
        
        # Generate random permutation for each sample
        # Using noise for sorting trick (more efficient than torch.randperm per sample)
        noise = torch.rand(batch_size, num_features, device=device)
        
        # Sort noise to get random ordering
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Split into visible and masked indices
        visible_indices = ids_shuffle[:, :num_visible]  # [batch, num_visible]
        masked_indices = ids_shuffle[:, num_visible:]   # [batch, num_masked]
        
        # Sort indices within each group for consistent ordering
        visible_indices = torch.sort(visible_indices, dim=1).values
        masked_indices = torch.sort(masked_indices, dim=1).values
        
        # Gather visible and masked tokens
        visible_tokens = self._gather_tokens(tokens, visible_indices)
        masked_tokens = self._gather_tokens(tokens, masked_indices)
        
        # Create boolean mask (True = visible)
        visible_mask = torch.zeros(batch_size, num_features, dtype=torch.bool, device=device)
        visible_mask.scatter_(1, visible_indices, True)
        
        return MaskingOutput(
            visible_indices=visible_indices,
            masked_indices=masked_indices,
            visible_tokens=visible_tokens,
            masked_tokens=masked_tokens,
            visible_mask=visible_mask,
        )
    
    def _gather_tokens(
        self, 
        tokens: torch.Tensor, 
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather tokens at specified indices."""
        batch_size, _, d_model = tokens.shape
        num_indices = indices.shape[1]
        
        # Expand indices for gathering: [batch, num_indices, d_model]
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
        
        return torch.gather(tokens, dim=1, index=indices_expanded)


class BlockMasking:
    """
    Mask contiguous blocks of features.
    
    Useful for structured tabular data where adjacent features may be related
    (e.g., features from the same category or measurement group).
    
    Args:
        num_blocks: Number of blocks to mask (default: 2)
        block_size: Size of each block (default: None, computed from ratio)
        mask_ratio: Target fraction of features to mask (default: 0.5)
        min_masked: Minimum number of features to mask
        min_visible: Minimum number of features to keep visible
    """
    
    def __init__(
        self,
        num_blocks: int = 2,
        block_size: Optional[int] = None,
        mask_ratio: float = 0.5,
        min_masked: int = 1,
        min_visible: int = 1,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.mask_ratio = mask_ratio
        self.min_masked = min_masked
        self.min_visible = min_visible
    
    def __call__(self, tokens: torch.Tensor) -> MaskingOutput:
        """
        Apply block masking to tokens.
        
        Args:
            tokens: [batch, num_features, d_model]
            
        Returns:
            MaskingOutput with visible/masked splits
        """
        batch_size, num_features, d_model = tokens.shape
        device = tokens.device
        
        # Compute block size if not specified
        if self.block_size is None:
            target_masked = int(num_features * self.mask_ratio)
            block_size = max(1, target_masked // self.num_blocks)
        else:
            block_size = self.block_size
        
        # Initialize mask (True = masked)
        mask = torch.zeros(batch_size, num_features, dtype=torch.bool, device=device)
        
        # For each sample, randomly place blocks
        for b in range(batch_size):
            masked_positions = set()
            
            for _ in range(self.num_blocks):
                # Random start position
                max_start = max(0, num_features - block_size)
                start = random.randint(0, max_start)
                
                # Add block positions
                for pos in range(start, min(start + block_size, num_features)):
                    masked_positions.add(pos)
            
            # Ensure min_visible constraint
            if len(masked_positions) > num_features - self.min_visible:
                masked_positions = set(random.sample(
                    list(masked_positions),
                    num_features - self.min_visible
                ))
            
            # Ensure min_masked constraint
            while len(masked_positions) < self.min_masked:
                remaining = set(range(num_features)) - masked_positions
                if remaining:
                    masked_positions.add(random.choice(list(remaining)))
                else:
                    break
            
            mask[b, list(masked_positions)] = True
        
        # Convert mask to indices
        visible_mask = ~mask
        
        # Get consistent counts - use the count from the first sample
        # (all samples should have valid counts due to constraints above)
        num_masked_per_sample = mask.sum(dim=1)  # [batch]
        num_visible_per_sample = visible_mask.sum(dim=1)  # [batch]
        
        # Use min to ensure we don't exceed any sample's actual count
        num_masked = int(num_masked_per_sample.min().item())
        num_visible = int(num_visible_per_sample.min().item())
        
        # Ensure we have at least min_masked and min_visible
        num_masked = max(num_masked, self.min_masked)
        num_visible = max(num_visible, self.min_visible)
        
        visible_indices = torch.zeros(batch_size, num_visible, dtype=torch.long, device=device)
        masked_indices = torch.zeros(batch_size, num_masked, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            vis_idx = torch.where(visible_mask[b])[0][:num_visible]
            msk_idx = torch.where(mask[b])[0][:num_masked]
            
            visible_indices[b, :len(vis_idx)] = vis_idx
            masked_indices[b, :len(msk_idx)] = msk_idx
        
        # Gather tokens
        visible_tokens = self._gather_tokens(tokens, visible_indices)
        masked_tokens = self._gather_tokens(tokens, masked_indices)
        
        return MaskingOutput(
            visible_indices=visible_indices,
            masked_indices=masked_indices,
            visible_tokens=visible_tokens,
            masked_tokens=masked_tokens,
            visible_mask=visible_mask,
        )
    
    def _gather_tokens(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather tokens at specified indices."""
        batch_size, _, d_model = tokens.shape
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
        return torch.gather(tokens, dim=1, index=indices_expanded)


class StructuredMasking:
    """
    Mask based on feature groups/types.
    
    Allows masking specific types of features (e.g., all categorical or
    all numerical) to encourage learning cross-feature relationships.
    
    Args:
        num_numerical: Number of numerical features (at start of sequence)
        num_categorical: Number of categorical features (after numerical)
        mask_ratio: Base mask ratio
        mask_numerical_prob: Probability of masking all numerical features
        mask_categorical_prob: Probability of masking all categorical features
    """
    
    def __init__(
        self,
        num_numerical: int,
        num_categorical: int,
        mask_ratio: float = 0.5,
        mask_numerical_prob: float = 0.2,
        mask_categorical_prob: float = 0.2,
        min_visible: int = 1,
    ):
        self.num_numerical = num_numerical
        self.num_categorical = num_categorical
        self.num_features = num_numerical + num_categorical
        self.mask_ratio = mask_ratio
        self.mask_numerical_prob = mask_numerical_prob
        self.mask_categorical_prob = mask_categorical_prob
        self.min_visible = min_visible
        
        # Fallback to random masking
        self.random_masking = RandomMasking(mask_ratio, min_visible=min_visible)
    
    def __call__(self, tokens: torch.Tensor) -> MaskingOutput:
        """
        Apply structured masking to tokens.
        
        Args:
            tokens: [batch, num_features, d_model]
            
        Returns:
            MaskingOutput with visible/masked splits
        """
        batch_size, num_features, d_model = tokens.shape
        device = tokens.device
        
        # Decide masking strategy
        rand = random.random()
        
        if rand < self.mask_numerical_prob and self.num_numerical > 0:
            # Mask all numerical features
            mask = torch.zeros(batch_size, num_features, dtype=torch.bool, device=device)
            mask[:, :self.num_numerical] = True
        elif rand < self.mask_numerical_prob + self.mask_categorical_prob and self.num_categorical > 0:
            # Mask all categorical features
            mask = torch.zeros(batch_size, num_features, dtype=torch.bool, device=device)
            mask[:, self.num_numerical:] = True
        else:
            # Fall back to random masking
            return self.random_masking(tokens)
        
        # Ensure min_visible
        if (~mask).sum(dim=1).min() < self.min_visible:
            return self.random_masking(tokens)
        
        # Build output
        visible_mask = ~mask
        
        visible_indices = torch.where(visible_mask[0])[0].unsqueeze(0).expand(batch_size, -1)
        masked_indices = torch.where(mask[0])[0].unsqueeze(0).expand(batch_size, -1)
        
        visible_tokens = self._gather_tokens(tokens, visible_indices)
        masked_tokens = self._gather_tokens(tokens, masked_indices)
        
        return MaskingOutput(
            visible_indices=visible_indices,
            masked_indices=masked_indices,
            visible_tokens=visible_tokens,
            masked_tokens=masked_tokens,
            visible_mask=visible_mask,
        )
    
    def _gather_tokens(
        self,
        tokens: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Gather tokens at specified indices."""
        batch_size, _, d_model = tokens.shape
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d_model)
        return torch.gather(tokens, dim=1, index=indices_expanded)


def create_masking_strategy(
    strategy: str = "random",
    mask_ratio: float = 0.5,
    **kwargs,
):
    """
    Factory function to create masking strategy.
    
    Args:
        strategy: Type of masking ("random", "block", "structured")
        mask_ratio: Fraction of features to mask
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Masking callable
    """
    if strategy == "random":
        return RandomMasking(mask_ratio=mask_ratio, **kwargs)
    elif strategy == "block":
        return BlockMasking(mask_ratio=mask_ratio, **kwargs)
    elif strategy == "structured":
        return StructuredMasking(mask_ratio=mask_ratio, **kwargs)
    else:
        raise ValueError(f"Unknown masking strategy: {strategy}")
