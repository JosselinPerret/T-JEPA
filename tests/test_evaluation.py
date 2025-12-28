"""Tests for evaluation module."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tempfile
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.jepa import TabularJEPA
from evaluation.linear_probe import LinearProbe, LinearProbeEvaluator, evaluate_linear_probe


def create_mock_dataloader(num_samples=100, num_numerical=5, num_categorical=3, num_classes=2, batch_size=16):
    """Create mock dataloaders for testing."""
    numerical_data = torch.randn(num_samples, num_numerical)
    categorical_data = torch.randint(0, 5, (num_samples, num_categorical))
    labels = torch.randint(0, num_classes, (num_samples,))
    
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, numerical, categorical, labels):
            self.numerical = numerical
            self.categorical = categorical
            self.labels = labels
        
        def __len__(self):
            return len(self.numerical)
        
        def __getitem__(self, idx):
            return {
                'numerical': self.numerical[idx],
                'categorical': self.categorical[idx],
                'label': self.labels[idx],
            }
    
    train_dataset = MockDataset(numerical_data[:80], categorical_data[:80], labels[:80])
    val_dataset = MockDataset(numerical_data[80:], categorical_data[80:], labels[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


def create_encoder():
    """Create a TabularJEPA encoder."""
    return TabularJEPA(
        num_numerical=5,
        category_sizes=[5, 5, 5],
        d_model=32,
        encoder_layers=2,
        encoder_heads=4,
        encoder_ff_dim=64,
        predictor_layers=2,
        predictor_heads=4,
        predictor_ff_dim=64,
        mask_ratio=0.3,
    )


class TestLinearProbe:
    """Tests for LinearProbe class."""
    
    def test_linear_probe_classification(self):
        """Test linear probe for classification."""
        probe = LinearProbe(
            input_dim=32,
            num_classes=2,
            task_type="classification",
            pooling="mean",
        )
        
        # Test forward pass
        x = torch.randn(16, 8, 32)  # [batch, tokens, dim]
        output = probe(x)
        
        assert output.shape == (16, 2)
    
    def test_linear_probe_regression(self):
        """Test linear probe for regression."""
        probe = LinearProbe(
            input_dim=32,
            num_classes=1,
            task_type="regression",
            pooling="mean",
        )
        
        x = torch.randn(16, 8, 32)
        output = probe(x)
        
        assert output.shape == (16, 1)
    
    def test_pooling_strategies(self):
        """Test different pooling strategies."""
        x = torch.randn(16, 8, 32)
        
        for pooling in ["mean", "first"]:
            probe = LinearProbe(
                input_dim=32,
                num_classes=2,
                task_type="classification",
                pooling=pooling,
            )
            
            output = probe(x)
            assert output.shape == (16, 2)


class TestLinearProbeEvaluator:
    """Tests for LinearProbeEvaluator class."""
    
    def test_evaluator_initialization(self):
        """Test evaluator can be initialized."""
        encoder = create_encoder()
        evaluator = LinearProbeEvaluator(encoder, device='cpu')
        
        assert evaluator.encoder is not None
        assert evaluator.device == 'cpu'
    
    def test_feature_extraction(self):
        """Test feature extraction."""
        encoder = create_encoder()
        evaluator = LinearProbeEvaluator(encoder, device='cpu')
        
        train_loader, _ = create_mock_dataloader()
        features, labels = evaluator.extract_features(train_loader)
        
        assert features.shape[0] == 80  # Number of samples
        assert features.shape[1] == 32  # d_model
        assert labels.shape[0] == 80
    
    def test_sklearn_classification(self):
        """Test sklearn linear probing for classification."""
        encoder = create_encoder()
        evaluator = LinearProbeEvaluator(encoder, device='cpu', mode='sklearn')
        
        train_loader, val_loader = create_mock_dataloader()
        
        metrics = evaluator.fit_sklearn(
            train_loader,
            val_loader,
            task_type="classification",
        )
        
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1
        assert 0 <= metrics['val_accuracy'] <= 1
    
    def test_sklearn_regression(self):
        """Test sklearn linear probing for regression."""
        encoder = create_encoder()
        evaluator = LinearProbeEvaluator(encoder, device='cpu', mode='sklearn')
        
        # Create regression data
        numerical_data = torch.randn(100, 5)
        categorical_data = torch.randint(0, 5, (100, 3))
        labels = torch.randn(100)
        
        class RegressionDataset(torch.utils.data.Dataset):
            def __init__(self, numerical, categorical, labels):
                self.numerical = numerical
                self.categorical = categorical
                self.labels = labels
            
            def __len__(self):
                return len(self.numerical)
            
            def __getitem__(self, idx):
                return {
                    'numerical': self.numerical[idx],
                    'categorical': self.categorical[idx],
                    'label': self.labels[idx],
                }
        
        train_dataset = RegressionDataset(numerical_data[:80], categorical_data[:80], labels[:80])
        val_dataset = RegressionDataset(numerical_data[80:], categorical_data[80:], labels[80:])
        
        train_loader = DataLoader(train_dataset, batch_size=16)
        val_loader = DataLoader(val_dataset, batch_size=16)
        
        metrics = evaluator.fit_sklearn(
            train_loader,
            val_loader,
            task_type="regression",
        )
        
        assert 'train_mse' in metrics
        assert 'train_r2' in metrics
        assert 'val_mse' in metrics
        assert 'val_r2' in metrics
    
    def test_pytorch_classification(self):
        """Test PyTorch linear probing."""
        encoder = create_encoder()
        evaluator = LinearProbeEvaluator(encoder, device='cpu', mode='pytorch')
        
        train_loader, val_loader = create_mock_dataloader()
        
        metrics = evaluator.fit_pytorch(
            train_loader,
            val_loader,
            num_classes=2,
            task_type="classification",
            epochs=2,
        )
        
        assert 'train_accuracy' in metrics
        assert 0 <= metrics['train_accuracy'] <= 1


class TestEvaluateFunction:
    """Tests for convenience function."""
    
    def test_evaluate_linear_probe_sklearn(self):
        """Test evaluate_linear_probe with sklearn mode."""
        encoder = create_encoder()
        train_loader, val_loader = create_mock_dataloader()
        
        metrics = evaluate_linear_probe(
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            num_classes=2,
            task_type="classification",
            mode="sklearn",
            device="cpu",
        )
        
        assert 'train_accuracy' in metrics
        assert 'val_accuracy' in metrics
        assert 'test_accuracy' in metrics
    
    def test_evaluate_linear_probe_pytorch(self):
        """Test evaluate_linear_probe with pytorch mode."""
        encoder = create_encoder()
        train_loader, val_loader = create_mock_dataloader()
        
        metrics = evaluate_linear_probe(
            encoder=encoder,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=None,
            num_classes=2,
            task_type="classification",
            mode="pytorch",
            device="cpu",
            epochs=2,
        )
        
        assert 'train_accuracy' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
