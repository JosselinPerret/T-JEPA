"""
Logging utilities for Tabular-JEPA.

Provides:
- Console logging with rich formatting
- TensorBoard logging
- Optional Weights & Biases integration
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

import torch


class Logger:
    """
    Unified logger for Tabular-JEPA training.
    
    Supports:
    - Console output with formatting
    - TensorBoard scalars and histograms
    - Optional W&B integration
    
    Args:
        log_dir: Directory for logs and tensorboard
        experiment_name: Name of the experiment
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        wandb_config: Config dict for W&B
    """
    
    def __init__(
        self,
        log_dir: str = "./outputs/logs",
        experiment_name: str = "tabular-jepa",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "tabular-jepa",
        wandb_config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        
        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup console logger
        self._setup_console_logger()
        
        # Setup TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            self._setup_tensorboard()
        
        # Setup W&B
        self.wandb_run = None
        if use_wandb:
            self._setup_wandb(wandb_project, wandb_config)
        
        self.step = 0
    
    def _setup_console_logger(self):
        """Setup Python logger for console output."""
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.run_dir / "train.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            self.logger.info(f"TensorBoard logs: {self.run_dir / 'tensorboard'}")
        except ImportError:
            self.logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            self.use_tensorboard = False
    
    def _setup_wandb(self, project: str, config: Optional[Dict]):
        """Setup Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=project,
                name=self.experiment_name,
                config=config,
                dir=str(self.run_dir),
            )
            self.logger.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            self.logger.warning("W&B not available. Install with: pip install wandb")
            self.use_wandb = False
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def log_scalar(
        self,
        tag: str,
        value: Union[float, torch.Tensor],
        step: Optional[int] = None,
    ):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the metric (e.g., 'train/loss')
            value: Scalar value
            step: Global step (uses internal counter if not provided)
        """
        if step is None:
            step = self.step
        
        if isinstance(value, torch.Tensor):
            value = value.item()
        
        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
        
        # W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: value}, step=step)
    
    def log_scalars(
        self,
        scalars: Dict[str, Union[float, torch.Tensor]],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log multiple scalars at once.
        
        Args:
            scalars: Dictionary of tag -> value
            step: Global step
            prefix: Optional prefix for all tags (e.g., 'train/')
        """
        if step is None:
            step = self.step
        
        for tag, value in scalars.items():
            full_tag = f"{prefix}{tag}" if prefix else tag
            self.log_scalar(full_tag, value, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None,
    ):
        """Log histogram of values."""
        if step is None:
            step = self.step
        
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values.detach().cpu(), step)
    
    def log_model_parameters(
        self,
        model: torch.nn.Module,
        step: Optional[int] = None,
    ):
        """Log model parameter histograms."""
        if step is None:
            step = self.step
        
        if self.tb_writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.tb_writer.add_histogram(f"params/{name}", param.detach().cpu(), step)
                    if param.grad is not None:
                        self.tb_writer.add_histogram(f"grads/{name}", param.grad.detach().cpu(), step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with final metrics."""
        if self.tb_writer is not None:
            self.tb_writer.add_hparams(hparams, metrics)
    
    def set_step(self, step: int):
        """Set the global step counter."""
        self.step = step
    
    def increment_step(self):
        """Increment the global step counter."""
        self.step += 1
    
    def close(self):
        """Close all writers and handlers."""
        # Close logging handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ProgressMeter:
    """
    Track and display training progress.
    
    Args:
        total_epochs: Total number of epochs
        steps_per_epoch: Steps per epoch
        logger: Logger instance
    """
    
    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        logger: Optional[Logger] = None,
    ):
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.logger = logger
        
        self.current_epoch = 0
        self.current_step = 0
        self.epoch_step = 0
        
        # Metrics tracking
        self.epoch_metrics: Dict[str, float] = {}
        self.epoch_counts: Dict[str, int] = {}
    
    def update(
        self,
        metrics: Dict[str, float],
        step: int = 1,
    ):
        """Update metrics for current step."""
        self.current_step += step
        self.epoch_step += step
        
        for name, value in metrics.items():
            if name not in self.epoch_metrics:
                self.epoch_metrics[name] = 0.0
                self.epoch_counts[name] = 0
            self.epoch_metrics[name] += value
            self.epoch_counts[name] += 1
    
    def get_epoch_average(self, name: str) -> float:
        """Get average of a metric over the current epoch."""
        if name not in self.epoch_metrics or self.epoch_counts[name] == 0:
            return 0.0
        return self.epoch_metrics[name] / self.epoch_counts[name]
    
    def get_all_epoch_averages(self) -> Dict[str, float]:
        """Get all epoch averages."""
        return {
            name: self.get_epoch_average(name)
            for name in self.epoch_metrics
        }
    
    def new_epoch(self, epoch: int):
        """Start a new epoch."""
        self.current_epoch = epoch
        self.epoch_step = 0
        self.epoch_metrics = {}
        self.epoch_counts = {}
    
    def log_step(self, log_interval: int = 100):
        """Log current step if at log interval."""
        if self.epoch_step % log_interval == 0:
            metrics_str = " | ".join(
                f"{k}: {self.get_epoch_average(k):.4f}"
                for k in self.epoch_metrics
            )
            progress = f"[{self.epoch_step}/{self.steps_per_epoch}]"
            message = f"Epoch {self.current_epoch} {progress} | {metrics_str}"
            
            if self.logger:
                self.logger.info(message)
            else:
                print(message)
    
    def log_epoch_summary(self):
        """Log end-of-epoch summary."""
        metrics_str = " | ".join(
            f"{k}: {self.get_epoch_average(k):.4f}"
            for k in self.epoch_metrics
        )
        message = f"Epoch {self.current_epoch} Complete | {metrics_str}"
        
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
