"""
Checkpointing utilities for Tabular-JEPA.

Provides saving and loading of:
- Model state
- Optimizer state
- Scheduler state
- Training state (epoch, step)
- Configuration
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import shutil

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Features:
    - Save/load complete training state
    - Keep best checkpoint based on metric
    - Keep last N checkpoints
    - Resume training from checkpoint
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep (0 = keep all)
        save_best: Whether to save best checkpoint
        best_metric: Metric name for determining best checkpoint
        best_mode: 'min' or 'max' for best metric comparison
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 3,
        save_best: bool = True,
        best_metric: str = "val_loss",
        best_mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.best_metric = best_metric
        self.best_mode = best_mode
        
        # Track checkpoints
        self.checkpoints = []
        self.best_value = float('inf') if best_mode == 'min' else float('-inf')
        self.best_checkpoint = None
    
    def save(
        self,
        epoch: int,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        ema_updater: Optional[Any] = None,
        config: Optional[DictConfig] = None,
        metrics: Optional[Dict[str, float]] = None,
        preprocessor_path: Optional[str] = None,
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            epoch: Current epoch
            step: Current global step
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Optional learning rate scheduler
            ema_updater: Optional EMA updater
            config: Optional configuration
            metrics: Optional current metrics
            preprocessor_path: Optional path to preprocessor
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if ema_updater is not None:
            checkpoint['ema_state_dict'] = ema_updater.state_dict()
        
        if config is not None:
            checkpoint['config'] = OmegaConf.to_container(config, resolve=True)
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        if preprocessor_path is not None:
            checkpoint['preprocessor_path'] = preprocessor_path
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch:04d}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        self.checkpoints.append(checkpoint_path)
        
        # Save best checkpoint
        if self.save_best and metrics is not None and self.best_metric in metrics:
            current_value = metrics[self.best_metric]
            is_best = (
                (self.best_mode == 'min' and current_value < self.best_value) or
                (self.best_mode == 'max' and current_value > self.best_value)
            )
            
            if is_best:
                self.best_value = current_value
                self.best_checkpoint = checkpoint_path
                best_path = self.checkpoint_dir / "checkpoint_best.pt"
                shutil.copy(checkpoint_path, best_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if self.max_checkpoints <= 0:
            return
        
        # Keep best checkpoint separate
        regular_checkpoints = [
            cp for cp in self.checkpoints
            if cp != self.best_checkpoint and cp.name != "checkpoint_best.pt"
        ]
        
        while len(regular_checkpoints) > self.max_checkpoints:
            oldest = regular_checkpoints.pop(0)
            if oldest.exists():
                oldest.unlink()
            if oldest in self.checkpoints:
                self.checkpoints.remove(oldest)
    
    def load(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        ema_updater: Optional[Any] = None,
        map_location: str = "cpu",
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            ema_updater: Optional EMA updater to load state into
            map_location: Device to map tensors to
            
        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Load model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA
        if ema_updater is not None and 'ema_state_dict' in checkpoint:
            ema_updater.load_state_dict(checkpoint['ema_state_dict'])
        
        return checkpoint
    
    def load_best(
        self,
        model: nn.Module,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best checkpoint found")
        return self.load(best_path, model, **kwargs)
    
    def load_latest(
        self,
        model: nn.Module,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if not checkpoints:
            raise FileNotFoundError("No checkpoints found")
        return self.load(checkpoints[-1], model, **kwargs)
    
    def get_resume_path(self) -> Optional[Path]:
        """Get path to resume from (latest checkpoint)."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        return checkpoints[-1] if checkpoints else None
    
    def exists(self) -> bool:
        """Check if any checkpoints exist."""
        return len(list(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))) > 0


def save_model_only(
    model: nn.Module,
    path: str,
    config: Optional[Dict] = None,
) -> None:
    """
    Save only the model (for inference/deployment).
    
    Args:
        model: Model to save
        path: Path to save file
        config: Optional model configuration
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if config is not None:
        save_dict['config'] = config
    
    torch.save(save_dict, path)


def load_model_only(
    model: nn.Module,
    path: str,
    map_location: str = "cpu",
    strict: bool = True,
) -> nn.Module:
    """
    Load only model weights.
    
    Args:
        model: Model to load weights into
        path: Path to saved model
        map_location: Device to map tensors to
        strict: Whether to require exact match of state dict keys
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=map_location)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    else:
        # Assume the file contains just the state dict
        model.load_state_dict(checkpoint, strict=strict)
    
    return model
