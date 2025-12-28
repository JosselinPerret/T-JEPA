"""
JEPA Loss functions for Tabular-JEPA.

Implements the loss functions for self-supervised pre-training:
- L2 Loss: MSE between predictions and targets in latent space
- VICReg: Variance-Invariance-Covariance regularization
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPALoss(nn.Module):
    """
    JEPA Loss: L2 distance in latent space between predictions and targets.
    
    The core JEPA objective is to minimize the distance between:
    - Predictor's output (what it thinks the target encoder would produce)
    - Target encoder's actual output (for masked features)
    
    Optional enhancements:
    - L2 normalization before loss computation
    - Smooth L1 loss (more robust to outliers)
    - VICReg regularization (prevents collapse)
    
    Args:
        normalize: Whether to L2-normalize embeddings before loss (default: True)
        loss_type: Type of loss ('mse', 'smooth_l1', 'cosine') (default: 'mse')
        vic_weight: Weight for VICReg regularization (0 = disabled) (default: 0.0)
        variance_weight: VICReg variance loss weight (default: 25.0)
        invariance_weight: VICReg invariance loss weight (default: 25.0)
        covariance_weight: VICReg covariance loss weight (default: 1.0)
    
    Example:
        >>> criterion = JEPALoss(normalize=True, loss_type='mse')
        >>> predictions = torch.randn(32, 4, 128)  # [batch, num_masked, d_model]
        >>> targets = torch.randn(32, 4, 128)
        >>> loss_dict = criterion(predictions, targets)
        >>> loss = loss_dict['loss']
        >>> loss.backward()
    """
    
    def __init__(
        self,
        normalize: bool = True,
        loss_type: str = "mse",
        vic_weight: float = 0.0,
        variance_weight: float = 25.0,
        invariance_weight: float = 25.0,
        covariance_weight: float = 1.0,
    ):
        super().__init__()
        self.normalize = normalize
        self.loss_type = loss_type
        self.vic_weight = vic_weight
        self.variance_weight = variance_weight
        self.invariance_weight = invariance_weight
        self.covariance_weight = covariance_weight
        
        # Validate loss type
        assert loss_type in ["mse", "smooth_l1", "cosine"], \
            f"Unknown loss type: {loss_type}"
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute JEPA loss.
        
        Args:
            predictions: [batch, num_masked, d_model] - predictor output
            targets: [batch, num_masked, d_model] - target encoder output
            
        Returns:
            Dictionary with:
                - 'loss': Total loss (scalar)
                - 'reconstruction_loss': L2/reconstruction loss
                - 'vic_loss': VICReg loss (if enabled)
        """
        batch_size, num_masked, d_model = predictions.shape
        
        # Optionally normalize embeddings
        if self.normalize:
            predictions = F.normalize(predictions, dim=-1, p=2)
            targets = F.normalize(targets, dim=-1, p=2)
        
        # Compute reconstruction loss
        if self.loss_type == "mse":
            reconstruction_loss = F.mse_loss(predictions, targets)
        elif self.loss_type == "smooth_l1":
            reconstruction_loss = F.smooth_l1_loss(predictions, targets)
        elif self.loss_type == "cosine":
            # 1 - cosine similarity (want to maximize similarity, minimize loss)
            cos_sim = F.cosine_similarity(
                predictions.reshape(-1, d_model),
                targets.reshape(-1, d_model),
                dim=-1
            )
            reconstruction_loss = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Build result dict
        result = {
            'reconstruction_loss': reconstruction_loss,
        }
        
        # Optional VICReg regularization
        if self.vic_weight > 0:
            vic_loss = self._vicreg_loss(predictions)
            result['vic_loss'] = vic_loss
            result['loss'] = reconstruction_loss + self.vic_weight * vic_loss
        else:
            result['vic_loss'] = torch.tensor(0.0, device=predictions.device)
            result['loss'] = reconstruction_loss
        
        return result
    
    def _vicreg_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute VICReg (Variance-Invariance-Covariance) regularization.
        
        Helps prevent representation collapse by:
        - Variance: Ensuring variance of embeddings doesn't collapse
        - Covariance: Decorrelating dimensions
        
        Args:
            z: [batch, num_masked, d_model] - embeddings to regularize
            
        Returns:
            VICReg loss (scalar)
        """
        batch_size, num_masked, d_model = z.shape
        
        # Flatten to [batch * num_masked, d_model]
        z_flat = z.reshape(-1, d_model)
        
        # Variance loss: encourage variance of each dimension to be >= 1
        std = torch.sqrt(z_flat.var(dim=0) + 1e-4)
        variance_loss = F.relu(1 - std).mean()
        
        # Covariance loss: decorrelate dimensions
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
        
        # Zero out diagonal (we only penalize off-diagonal covariance)
        cov_off_diag = cov - torch.diag(torch.diag(cov))
        covariance_loss = (cov_off_diag ** 2).sum() / d_model
        
        # Combined VICReg loss (no invariance term here as it's handled by main loss)
        vic_loss = (
            self.variance_weight * variance_loss +
            self.covariance_weight * covariance_loss
        )
        
        return vic_loss
    
    def extra_repr(self) -> str:
        return (
            f"normalize={self.normalize}, "
            f"loss_type={self.loss_type}, "
            f"vic_weight={self.vic_weight}"
        )


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss (alternative to L2 loss).
    
    Treats correct prediction-target pairs as positives and
    other samples in the batch as negatives.
    
    Args:
        temperature: Temperature for softmax (default: 0.1)
        normalize: Whether to normalize embeddings (default: True)
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        normalize: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute InfoNCE loss.
        
        Args:
            predictions: [batch, num_masked, d_model]
            targets: [batch, num_masked, d_model]
            
        Returns:
            Dictionary with 'loss' key
        """
        batch_size, num_masked, d_model = predictions.shape
        
        # Flatten: [batch * num_masked, d_model]
        pred_flat = predictions.reshape(-1, d_model)
        tgt_flat = targets.reshape(-1, d_model)
        
        if self.normalize:
            pred_flat = F.normalize(pred_flat, dim=-1, p=2)
            tgt_flat = F.normalize(tgt_flat, dim=-1, p=2)
        
        # Compute similarity matrix: [N, N] where N = batch * num_masked
        sim_matrix = torch.mm(pred_flat, tgt_flat.T) / self.temperature
        
        # Labels: diagonal elements are positives
        labels = torch.arange(sim_matrix.shape[0], device=sim_matrix.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        # Compute accuracy for logging
        with torch.no_grad():
            accuracy = (sim_matrix.argmax(dim=-1) == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy,
        }


def create_loss_function(
    loss_type: str = "mse",
    normalize: bool = True,
    vic_weight: float = 0.0,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: Type of loss ('mse', 'smooth_l1', 'cosine', 'infonce')
        normalize: Whether to normalize embeddings
        vic_weight: VICReg weight
        **kwargs: Additional loss-specific arguments
        
    Returns:
        Loss module
    """
    if loss_type == "infonce":
        return InfoNCELoss(normalize=normalize, **kwargs)
    else:
        return JEPALoss(
            loss_type=loss_type,
            normalize=normalize,
            vic_weight=vic_weight,
            **kwargs,
        )
