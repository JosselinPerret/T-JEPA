"""
Exponential Moving Average (EMA) updater for Tabular-JEPA.

Handles the EMA updates for the target encoder with optional
scheduling (warmup, cosine decay, etc.).
"""

import math
from typing import Iterator, Optional

import torch
import torch.nn as nn


class EMAUpdater:
    """
    Handles EMA updates for the target encoder with scheduling.
    
    The target encoder's weights are updated as:
        θ_target = decay * θ_target + (1 - decay) * θ_online
    
    Scheduling options:
        - Constant: Fixed decay throughout training
        - Warmup: Linear warmup from base_ema to target decay
        - Cosine: Cosine schedule from base_ema to max_ema
    
    Args:
        base_ema: Starting EMA decay (default: 0.996)
        max_ema: Maximum EMA decay at end of schedule (default: 1.0)
        warmup_steps: Number of warmup steps (default: 0)
        total_steps: Total training steps for scheduling (default: 100000)
        schedule: Type of schedule ('constant', 'warmup', 'cosine')
    
    Example:
        >>> ema = EMAUpdater(base_ema=0.996, max_ema=1.0, total_steps=10000)
        >>> for step in range(10000):
        ...     # Training step
        ...     ema.update(context_encoder.parameters(), target_encoder.parameters())
        ...     current_decay = ema.get_ema_decay()
    """
    
    def __init__(
        self,
        base_ema: float = 0.996,
        max_ema: float = 1.0,
        warmup_steps: int = 0,
        total_steps: int = 100000,
        schedule: str = "cosine",
    ):
        self.base_ema = base_ema
        self.max_ema = max_ema
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.schedule = schedule
        self.step = 0
        
        # Validate
        assert 0 <= base_ema <= 1, "base_ema must be in [0, 1]"
        assert 0 <= max_ema <= 1, "max_ema must be in [0, 1]"
        assert base_ema <= max_ema, "base_ema must be <= max_ema"
    
    def get_ema_decay(self) -> float:
        """
        Get current EMA decay based on step and schedule.
        
        Returns:
            Current EMA decay value
        """
        if self.schedule == "constant":
            return self.base_ema
        
        elif self.schedule == "warmup":
            if self.step < self.warmup_steps:
                # Linear warmup
                return self.base_ema
            else:
                # After warmup, use max_ema
                return self.max_ema
        
        elif self.schedule == "cosine":
            # Cosine schedule from base_ema to max_ema
            if self.step >= self.total_steps:
                return self.max_ema
            
            # Cosine annealing
            progress = self.step / self.total_steps
            decay = self.base_ema + (self.max_ema - self.base_ema) * (
                1 - math.cos(math.pi * progress)
            ) / 2
            return decay
        
        elif self.schedule == "linear":
            # Linear schedule from base_ema to max_ema
            if self.step >= self.total_steps:
                return self.max_ema
            
            progress = self.step / self.total_steps
            decay = self.base_ema + (self.max_ema - self.base_ema) * progress
            return decay
        
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")
    
    @torch.no_grad()
    def update(
        self,
        online_params: Iterator[nn.Parameter],
        target_params: Iterator[nn.Parameter],
        step: Optional[int] = None,
    ) -> float:
        """
        Perform EMA update step.
        
        Args:
            online_params: Parameters from the online (context) encoder
            target_params: Parameters from the target encoder
            step: Optional step override (otherwise uses internal counter)
            
        Returns:
            The decay value used for this update
        """
        if step is not None:
            self.step = step
        
        decay = self.get_ema_decay()
        
        for param_online, param_target in zip(online_params, target_params):
            param_target.data.mul_(decay).add_(param_online.data, alpha=1 - decay)
        
        self.step += 1
        return decay
    
    def update_model(
        self,
        online_model: nn.Module,
        target_model: nn.Module,
        step: Optional[int] = None,
    ) -> float:
        """
        Convenience method to update target model from online model.
        
        Args:
            online_model: The online (context) encoder
            target_model: The target encoder
            step: Optional step override
            
        Returns:
            The decay value used for this update
        """
        return self.update(
            online_model.parameters(),
            target_model.parameters(),
            step=step,
        )
    
    def reset(self) -> None:
        """Reset the step counter."""
        self.step = 0
    
    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            'step': self.step,
            'base_ema': self.base_ema,
            'max_ema': self.max_ema,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'schedule': self.schedule,
        }
    
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.step = state_dict['step']
        self.base_ema = state_dict['base_ema']
        self.max_ema = state_dict['max_ema']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.schedule = state_dict['schedule']
    
    def __repr__(self) -> str:
        return (
            f"EMAUpdater(base_ema={self.base_ema}, max_ema={self.max_ema}, "
            f"schedule={self.schedule}, step={self.step}/{self.total_steps})"
        )


def copy_params(source: nn.Module, target: nn.Module) -> None:
    """
    Copy parameters from source to target model.
    
    Useful for initializing target encoder from context encoder.
    
    Args:
        source: Source model to copy from
        target: Target model to copy to
    """
    for param_source, param_target in zip(source.parameters(), target.parameters()):
        param_target.data.copy_(param_source.data)
