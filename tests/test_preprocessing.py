"""
Unit tests for data preprocessing and dataset classes.

Run with: python tests/test_preprocessing.py
Or with pytest: python -m pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create dummy fixtures for running without pytest
    class DummyFixture:
        def __call__(self, func):
            return func
    class DummyPytest:
        fixture = DummyFixture()
    pytest = DummyPytest()

from data.preprocessing import TabularPreprocessor, ColumnInfo
from data.datasets import TabularDataset


class TestTabularPreprocessor:
    """Tests for TabularPreprocessor class."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create a sample mixed-type DataFrame for testing."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
            'age': np.random.randint(18, 80, n),
            'income': np.random.randn(n) * 10000 + 50000,
            'hours_per_week': np.random.randint(10, 60, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], n),
            'occupation': np.random.choice(['Tech', 'Sales', 'Admin', 'Other'], n),
            'target': np.random.choice([0, 1], n),
        })
    
    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = TabularPreprocessor()
        assert preprocessor._is_fitted == False
        assert preprocessor.numerical_cols == []
        assert preprocessor.categorical_cols == []
    
    def test_auto_detection(self, sample_dataframe):
        """Test automatic column type detection."""
        preprocessor = TabularPreprocessor(auto_detect=True)
        preprocessor.fit(sample_dataframe, target_col='target')
        
        # Check that columns were detected
        assert len(preprocessor.numerical_cols) > 0
        assert len(preprocessor.categorical_cols) > 0
        assert 'income' in preprocessor.numerical_cols
        assert 'gender' in preprocessor.categorical_cols
        assert 'target' not in preprocessor.feature_order
    
    def test_manual_columns(self, sample_dataframe):
        """Test with manually specified columns."""
        preprocessor = TabularPreprocessor(
            numerical_cols=['age', 'income'],
            categorical_cols=['gender', 'education'],
            auto_detect=False,
        )
        preprocessor.fit(sample_dataframe, target_col='target')
        
        assert preprocessor.numerical_cols == ['age', 'income']
        assert preprocessor.categorical_cols == ['gender', 'education']
    
    def test_transform_shapes(self, sample_dataframe):
        """Test that transform produces correct tensor shapes."""
        preprocessor = TabularPreprocessor()
        result = preprocessor.fit_transform(sample_dataframe, target_col='target')
        
        n = len(sample_dataframe)
        
        assert 'numerical' in result
        assert 'categorical' in result
        assert 'labels' in result
        
        assert result['numerical'].shape[0] == n
        assert result['categorical'].shape[0] == n
        assert result['labels'].shape[0] == n
        
        assert result['numerical'].dtype == torch.float32
        assert result['categorical'].dtype == torch.int64
        assert result['labels'].dtype == torch.int64
    
    def test_transform_normalization(self, sample_dataframe):
        """Test that numerical features are normalized."""
        preprocessor = TabularPreprocessor()
        result = preprocessor.fit_transform(sample_dataframe, target_col='target')
        
        numerical = result['numerical']
        
        # Check approximate normalization (mean ≈ 0, std ≈ 1)
        for i in range(numerical.shape[1]):
            col_mean = numerical[:, i].mean().item()
            col_std = numerical[:, i].std().item()
            assert abs(col_mean) < 0.2, f"Column {i} mean not near 0: {col_mean}"
            assert 0.8 < col_std < 1.2, f"Column {i} std not near 1: {col_std}"
    
    def test_categorical_encoding(self, sample_dataframe):
        """Test that categorical features are properly encoded."""
        preprocessor = TabularPreprocessor()
        result = preprocessor.fit_transform(sample_dataframe, target_col='target')
        
        categorical = result['categorical']
        
        # All values should be non-negative (0 = unknown/padding)
        assert (categorical >= 0).all()
        
        # Values should be within vocab size
        for i, size in enumerate(preprocessor.category_sizes):
            assert categorical[:, i].max() <= size
    
    def test_unknown_categories(self, sample_dataframe):
        """Test handling of unknown categories at transform time."""
        preprocessor = TabularPreprocessor()
        preprocessor.fit(sample_dataframe, target_col='target')
        
        # Create new data with unknown category
        new_df = sample_dataframe.copy()
        new_df.loc[0, 'gender'] = 'Unknown'  # New category not in training
        
        result = preprocessor.transform(new_df)
        
        # Unknown should be encoded as 0 (padding index)
        # This is expected behavior
        assert result['categorical'].shape[0] == len(new_df)
    
    def test_missing_values(self, sample_dataframe):
        """Test handling of missing values."""
        df = sample_dataframe.copy()
        df.loc[0, 'age'] = np.nan
        df.loc[1, 'income'] = np.nan
        df.loc[2, 'gender'] = np.nan
        
        preprocessor = TabularPreprocessor()
        result = preprocessor.fit_transform(df, target_col='target')
        
        # Should not have NaN in output
        assert not torch.isnan(result['numerical']).any()
    
    def test_get_config(self, sample_dataframe):
        """Test configuration export."""
        preprocessor = TabularPreprocessor()
        preprocessor.fit(sample_dataframe, target_col='target')
        
        config = preprocessor.get_config()
        
        assert 'num_numerical' in config
        assert 'num_categorical' in config
        assert 'category_sizes' in config
        assert 'feature_order' in config
        assert 'num_classes' in config
        
        assert config['num_features'] == config['num_numerical'] + config['num_categorical']
    
    def test_save_load(self, sample_dataframe, tmp_path):
        """Test serialization and deserialization."""
        preprocessor = TabularPreprocessor()
        preprocessor.fit(sample_dataframe, target_col='target')
        
        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor.save(str(save_path))
        
        # Load
        loaded = TabularPreprocessor.load(str(save_path))
        
        # Compare configs
        assert preprocessor.get_config() == loaded.get_config()
        
        # Compare transforms
        result1 = preprocessor.transform(sample_dataframe)
        result2 = loaded.transform(sample_dataframe)
        
        assert torch.allclose(result1['numerical'], result2['numerical'])
        assert torch.equal(result1['categorical'], result2['categorical'])


class TestTabularDataset:
    """Tests for TabularDataset class."""
    
    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        n = 50
        return {
            'numerical': torch.randn(n, 5),
            'categorical': torch.randint(0, 10, (n, 3)),
            'labels': torch.randint(0, 2, (n,)),
        }
    
    def test_dataset_length(self, sample_tensors):
        """Test dataset length."""
        dataset = TabularDataset(sample_tensors)
        assert len(dataset) == 50
    
    def test_dataset_getitem(self, sample_tensors):
        """Test dataset indexing."""
        dataset = TabularDataset(sample_tensors)
        item = dataset[0]
        
        assert 'numerical' in item
        assert 'categorical' in item
        assert 'labels' in item
        
        assert item['numerical'].shape == (5,)
        assert item['categorical'].shape == (3,)
        assert item['labels'].shape == ()
    
    def test_dataset_without_labels(self, sample_tensors):
        """Test dataset without labels."""
        dataset = TabularDataset(sample_tensors, include_labels=False)
        item = dataset[0]
        
        assert 'labels' not in item
    
    def test_dataloader_integration(self, sample_tensors):
        """Test DataLoader integration."""
        from torch.utils.data import DataLoader
        
        dataset = TabularDataset(sample_tensors)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        batch = next(iter(loader))
        
        assert batch['numerical'].shape == (16, 5)
        assert batch['categorical'].shape == (16, 3)
        assert batch['labels'].shape == (16,)


def test_full_pipeline():
    """Integration test for full data pipeline."""
    # Create synthetic data
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'num1': np.random.randn(n),
        'num2': np.random.randn(n) * 10,
        'cat1': np.random.choice(['A', 'B', 'C'], n),
        'cat2': np.random.choice(['X', 'Y'], n),
        'label': np.random.choice([0, 1], n),
    })
    
    # Split
    train_df = df.iloc[:160]
    test_df = df.iloc[160:]
    
    # Preprocess
    preprocessor = TabularPreprocessor(
        numerical_cols=['num1', 'num2'],
        categorical_cols=['cat1', 'cat2'],
        auto_detect=False,
    )
    
    train_data = preprocessor.fit_transform(train_df, target_col='label')
    test_data = preprocessor.transform(test_df)
    
    # Create datasets
    train_ds = TabularDataset(train_data)
    test_ds = TabularDataset(test_data)
    
    # Create loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    # Verify batch
    batch = next(iter(train_loader))
    assert batch['numerical'].shape == (32, 2)
    assert batch['categorical'].shape == (32, 2)
    
    # Get model config
    config = preprocessor.get_config()
    assert config['num_numerical'] == 2
    assert config['num_categorical'] == 2
    assert config['num_features'] == 4
    
    print("✓ Full pipeline test passed!")


if __name__ == "__main__":
    # Run tests without pytest
    print("Running data pipeline tests...\n")
    
    test_full_pipeline()
    
    # Quick manual tests
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.randn(100) * 10000 + 50000,
        'gender': np.random.choice(['Male', 'Female'], 100),
        'education': np.random.choice(['HS', 'BS', 'MS', 'PhD'], 100),
        'target': np.random.choice([0, 1], 100),
    })
    
    preprocessor = TabularPreprocessor()
    data = preprocessor.fit_transform(df, target_col='target')
    
    print(f"\nPreprocessor: {preprocessor}")
    print(f"Config: {preprocessor.get_config()}")
    print(f"Numerical shape: {data['numerical'].shape}")
    print(f"Categorical shape: {data['categorical'].shape}")
    print(f"Labels shape: {data['labels'].shape}")
    
    print("\n✓ All manual tests passed!")
