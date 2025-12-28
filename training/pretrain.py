"""
Self-supervised pre-training loop for Tabular-JEPA.

Implements the complete training pipeline:
- Data loading and preprocessing
- Model initialization
- Training loop with gradient accumulation
- EMA updates
- Validation
- Checkpointing
- Logging
"""

import math
import time
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from omegaconf import DictConfig
from tqdm import tqdm

from models.jepa import TabularJEPA
from losses.jepa_loss import JEPALoss, create_loss_function
from training.ema import EMAUpdater
from utils.logging import Logger, ProgressMeter
from utils.checkpointing import CheckpointManager
from utils.config import get_device


class Trainer:
    """
    Trainer class for Tabular-JEPA self-supervised pre-training.
    
    Handles:
    - Training loop
    - Validation
    - EMA updates
    - Learning rate scheduling
    - Checkpointing
    - Logging
    
    Args:
        model: TabularJEPA model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        logger: Logger instance
        checkpoint_manager: Checkpoint manager
        device: Device to train on
    """
    
    def __init__(
        self,
        model: TabularJEPA,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: DictConfig,
        logger: Logger,
        checkpoint_manager: CheckpointManager,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = create_loss_function(
            loss_type=config.loss.loss_type,
            normalize=config.loss.normalize,
            vic_weight=config.loss.vic_weight,
        )
        
        # Setup EMA updater
        total_steps = len(train_loader) * config.pretrain.epochs
        self.ema_updater = EMAUpdater(
            base_ema=config.jepa.ema_decay_base,
            max_ema=config.jepa.ema_decay_max,
            warmup_steps=config.jepa.ema_warmup_steps,
            total_steps=total_steps,
            schedule=config.jepa.ema_schedule,
        )
        
        # Mixed precision
        self.use_amp = device == "cuda"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        cfg = self.config.pretrain
        
        # Get parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        if cfg.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=cfg.learning_rate,
                betas=tuple(cfg.betas),
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=cfg.learning_rate,
                betas=tuple(cfg.betas),
            )
        elif cfg.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=cfg.learning_rate,
                momentum=0.9,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler based on config."""
        cfg = self.config.pretrain
        total_steps = len(self.train_loader) * cfg.epochs
        warmup_steps = len(self.train_loader) * cfg.warmup_epochs
        
        # Avoid division by zero when total_steps == warmup_steps
        decay_steps = max(total_steps - warmup_steps, 1)
        
        if cfg.scheduler == "cosine":
            # Cosine annealing with warmup
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                else:
                    progress = (step - warmup_steps) / decay_steps
                    return cfg.min_lr / cfg.learning_rate + (1 - cfg.min_lr / cfg.learning_rate) * (
                        0.5 * (1 + math.cos(math.pi * progress))
                    )
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif cfg.scheduler == "linear":
            # Linear warmup then linear decay
            def lr_lambda(step):
                if warmup_steps > 0 and step < warmup_steps:
                    return step / warmup_steps
                else:
                    return max(
                        cfg.min_lr / cfg.learning_rate,
                        1 - (step - warmup_steps) / decay_steps
                    )
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif cfg.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(cfg.epochs // 3, 1),
                gamma=0.1,
            )
        
        elif cfg.scheduler == "none":
            scheduler = None
        
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()
        self.current_epoch = epoch
        
        progress = ProgressMeter(
            total_epochs=self.config.pretrain.epochs,
            steps_per_epoch=len(self.train_loader),
            logger=self.logger,
        )
        progress.new_epoch(epoch)
        
        accumulation_steps = self.config.pretrain.gradient_accumulation_steps
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            numerical = batch['numerical'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            
            # Forward pass with optional mixed precision
            if self.use_amp:
                with autocast():
                    output = self.model(numerical, categorical)
                    loss_dict = self.criterion(output['predictions'], output['targets'])
                    loss = loss_dict['loss'] / accumulation_steps
            else:
                output = self.model(numerical, categorical)
                loss_dict = self.criterion(output['predictions'], output['targets'])
                loss = loss_dict['loss'] / accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.config.pretrain.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.pretrain.gradient_clip
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # EMA update
                self.ema_updater.update(
                    self.model.context_encoder.parameters(),
                    self.model.target_encoder.parameters(),
                )
                
                self.global_step += 1
            
            # Track metrics
            metrics = {
                'loss': loss_dict['loss'].item(),
                'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
            }
            if 'vic_loss' in loss_dict:
                metrics['vic_loss'] = loss_dict['vic_loss'].item()
            
            progress.update(metrics)
            
            # Logging
            if batch_idx % self.config.pretrain.log_interval == 0:
                progress.log_step(self.config.pretrain.log_interval)
                
                # Log to TensorBoard/W&B
                lr = self.optimizer.param_groups[0]['lr']
                ema_decay = self.ema_updater.get_ema_decay()
                
                self.logger.log_scalars({
                    'loss': loss_dict['loss'].item(),
                    'reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                    'lr': lr,
                    'ema_decay': ema_decay,
                }, step=self.global_step, prefix='train/')
        
        # End of epoch
        epoch_metrics = progress.get_all_epoch_averages()
        progress.log_epoch_summary()
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation.
        
        Returns:
            Dictionary of average validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_recon_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            numerical = batch['numerical'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            
            output = self.model(numerical, categorical)
            loss_dict = self.criterion(output['predictions'], output['targets'])
            
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_reconstruction_loss': total_recon_loss / num_batches,
        }
        
        # Log validation metrics
        self.logger.log_scalars(metrics, step=self.global_step, prefix='')
        self.logger.info(f"Validation | loss: {metrics['val_loss']:.4f}")
        
        return metrics
    
    def train(self) -> Dict[str, float]:
        """
        Run full training loop.
        
        Returns:
            Final metrics dictionary
        """
        self.logger.info(f"Starting training for {self.config.pretrain.epochs} epochs")
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Total parameters: {self.model.get_num_parameters(trainable_only=False):,}")
        self.logger.info(f"Trainable parameters: {self.model.get_num_parameters(trainable_only=True):,}")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.pretrain.epochs + 1):
            epoch_start = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Epoch timing
            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.1f}s")
            
            # Checkpointing
            if epoch % self.config.pretrain.save_interval == 0:
                all_metrics = {**train_metrics, **val_metrics}
                self.checkpoint_manager.save(
                    epoch=epoch,
                    step=self.global_step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    ema_updater=self.ema_updater,
                    config=self.config,
                    metrics=all_metrics,
                )
                self.logger.info(f"Saved checkpoint at epoch {epoch}")
            
            # Track best
            if 'val_loss' in val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time / 60:.1f} minutes")
        
        return {
            'final_train_loss': train_metrics.get('loss', 0),
            'final_val_loss': val_metrics.get('val_loss', 0),
            'best_val_loss': self.best_val_loss,
        }
    
    def resume(self, checkpoint_path: str) -> int:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Epoch to resume from
        """
        checkpoint = self.checkpoint_manager.load(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_updater=self.ema_updater,
            map_location=self.device,
        )
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        
        self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
        
        return self.current_epoch


def create_trainer(
    model: TabularJEPA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    config: DictConfig,
    output_dir: str,
) -> Trainer:
    """
    Factory function to create a Trainer with all dependencies.
    
    Args:
        model: TabularJEPA model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration
        output_dir: Output directory for logs and checkpoints
        
    Returns:
        Configured Trainer instance
    """
    output_dir = Path(output_dir)
    
    # Create logger
    logger = Logger(
        log_dir=str(output_dir / "logs"),
        experiment_name=config.experiment.name,
        use_tensorboard=config.experiment.use_tensorboard,
        use_wandb=config.experiment.use_wandb,
        wandb_project=config.experiment.wandb_project,
    )
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints"),
        max_checkpoints=3,
        save_best=True,
        best_metric="val_loss",
        best_mode="min",
    )
    
    # Get device
    device = get_device(config.experiment.device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device,
    )
    
    return trainer
