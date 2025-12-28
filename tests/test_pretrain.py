"""Tests for the pre-training loop."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import TabularJEPA
from training.pretrain import Trainer, create_trainer
from training.ema import EMAUpdater
from utils.logging import Logger
from utils.checkpointing import CheckpointManager


def create_mock_config():
    """Create a mock configuration for testing."""
    config = OmegaConf.create({
        'data': {
            'dataset_name': 'test',
            'data_dir': './data',
            'val_split': 0.1,
            'test_split': 0.1,
        },
        'model': {
            'd_model': 32,
            'encoder_layers': 2,
            'encoder_heads': 4,
            'encoder_ff_dim': 64,
            'dropout': 0.1,
        },
        'jepa': {
            'mask_type': 'random',
            'mask_ratio': 0.3,
            'predictor_layers': 2,
            'predictor_heads': 4,
            'predictor_ff_dim': 64,
            'predictor_type': 'transformer',
            'ema_decay': 0.99,
            'ema_decay_base': 0.99,
            'ema_decay_max': 0.999,
            'ema_warmup_steps': 10,
            'ema_schedule': 'constant',
        },
        'loss': {
            'loss_type': 'mse',
            'normalize': False,
            'vic_weight': 0.0,
        },
        'pretrain': {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'betas': [0.9, 0.999],
            'weight_decay': 0.05,
            'warmup_epochs': 1,
            'min_lr': 1e-5,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'gradient_accumulation_steps': 1,
            'gradient_clip': 1.0,
            'log_interval': 1,
            'save_interval': 1,
            'num_workers': 0,
        },
        'experiment': {
            'name': 'test_pretrain',
            'seed': 42,
            'device': 'cpu',
            'use_tensorboard': False,
            'use_wandb': False,
            'wandb_project': '',
        },
    })
    return config


def create_mock_dataloader(num_samples=100, num_numerical=5, num_categorical=3, batch_size=16):
    """Create mock dataloaders for testing."""
    # Create mock data
    numerical_data = torch.randn(num_samples, num_numerical)
    categorical_data = torch.randint(0, 5, (num_samples, num_categorical))
    
    # Create dataset
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, numerical, categorical):
            self.numerical = numerical
            self.categorical = categorical
        
        def __len__(self):
            return len(self.numerical)
        
        def __getitem__(self, idx):
            return {
                'numerical': self.numerical[idx],
                'categorical': self.categorical[idx],
            }
    
    train_dataset = MockDataset(numerical_data[:80], categorical_data[:80])
    val_dataset = MockDataset(numerical_data[80:], categorical_data[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_model(config):
    """Create a TabularJEPA model from config."""
    return TabularJEPA(
        num_numerical=5,
        category_sizes=[5, 5, 5],
        d_model=config.model.d_model,
        encoder_layers=config.model.encoder_layers,
        encoder_heads=config.model.encoder_heads,
        encoder_ff_dim=config.model.encoder_ff_dim,
        predictor_layers=config.jepa.predictor_layers,
        predictor_heads=config.jepa.predictor_heads,
        predictor_ff_dim=config.jepa.predictor_ff_dim,
        dropout=config.model.dropout,
        mask_ratio=config.jepa.mask_ratio,
        mask_type=config.jepa.mask_type,
        predictor_type=config.jepa.predictor_type,
        ema_decay=config.jepa.ema_decay,
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for tests."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup after test - use ignore_errors for Windows compatibility
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestTrainer:
    """Test class for Trainer."""
    
    def test_trainer_initialization(self, temp_output_dir):
        """Test trainer can be initialized."""
        config = create_mock_config()
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            assert trainer.criterion is not None
        finally:
            logger.close()
    
    def test_train_one_epoch(self, temp_output_dir):
        """Test training for one epoch."""
        config = create_mock_config()
        config.pretrain.epochs = 1
        
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            metrics = trainer.train_epoch(epoch=1)
            
            assert 'loss' in metrics
            assert metrics['loss'] > 0
        finally:
            logger.close()
    
    def test_validation(self, temp_output_dir):
        """Test validation loop."""
        config = create_mock_config()
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            val_metrics = trainer.validate()
            
            assert 'val_loss' in val_metrics
            assert val_metrics['val_loss'] > 0
        finally:
            logger.close()
    
    def test_full_training_loop(self, temp_output_dir):
        """Test full training loop."""
        config = create_mock_config()
        config.pretrain.epochs = 2
        
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            final_metrics = trainer.train()
            
            assert 'final_train_loss' in final_metrics
            assert 'final_val_loss' in final_metrics
            assert 'best_val_loss' in final_metrics
        finally:
            logger.close()
    
    def test_ema_update(self, temp_output_dir):
        """Test that EMA updates the target encoder."""
        config = create_mock_config()
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        # Get initial target encoder params
        initial_target_params = [p.clone() for p in model.target_encoder.parameters()]
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            # Train for one epoch
            trainer.train_epoch(epoch=1)
            
            # Check that target encoder params have changed
            changed = False
            for old_p, new_p in zip(initial_target_params, model.target_encoder.parameters()):
                if not torch.allclose(old_p, new_p):
                    changed = True
                    break
            
            assert changed, "Target encoder should be updated via EMA"
        finally:
            logger.close()
    
    def test_gradient_accumulation(self, temp_output_dir):
        """Test gradient accumulation."""
        config = create_mock_config()
        config.pretrain.gradient_accumulation_steps = 2
        
        train_loader, val_loader = create_mock_dataloader(batch_size=8)
        model = create_model(config)
        
        logger = Logger(
            log_dir=temp_output_dir,
            experiment_name='test',
            use_tensorboard=False,
            use_wandb=False,
        )
        checkpoint_manager = CheckpointManager(checkpoint_dir=temp_output_dir)
        
        try:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                logger=logger,
                checkpoint_manager=checkpoint_manager,
                device='cpu',
            )
            
            # Should not error with gradient accumulation
            metrics = trainer.train_epoch(epoch=1)
            assert 'loss' in metrics
        finally:
            logger.close()


class TestCreateTrainer:
    """Test factory function."""
    
    def test_create_trainer(self, temp_output_dir):
        """Test create_trainer factory function."""
        config = create_mock_config()
        train_loader, val_loader = create_mock_dataloader()
        model = create_model(config)
        
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=temp_output_dir,
        )
        
        try:
            assert isinstance(trainer, Trainer)
            assert trainer.model is not None
        finally:
            trainer.logger.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
