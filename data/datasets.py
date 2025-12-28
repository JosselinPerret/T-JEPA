"""
PyTorch Dataset classes for Tabular-JEPA.

Provides efficient data loading for self-supervised pre-training
and downstream classification tasks.
"""

from typing import Dict, Optional, Tuple, List, Any, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import openml

from .preprocessing import TabularPreprocessor


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    
    Stores preprocessed tensors in memory for fast access during training.
    Supports both pre-training (no labels) and fine-tuning (with labels).
    
    Example:
        >>> preprocessor = TabularPreprocessor()
        >>> train_tensors = preprocessor.fit_transform(train_df, target_col='label')
        >>> dataset = TabularDataset(train_tensors)
        >>> loader = DataLoader(dataset, batch_size=256, shuffle=True)
    """
    
    def __init__(
        self,
        data: Dict[str, torch.Tensor],
        include_labels: bool = True,
    ):
        """
        Args:
            data: Dictionary from TabularPreprocessor.transform() containing:
                - 'numerical': Tensor[N, num_numerical]
                - 'categorical': Tensor[N, num_categorical]
                - 'labels': Optional Tensor[N]
            include_labels: Whether to include labels in __getitem__
        """
        self.numerical = data['numerical']
        self.categorical = data['categorical']
        self.labels = data.get('labels', None)
        self.include_labels = include_labels and self.labels is not None
        
        self.num_samples = self.numerical.shape[0]
        self.num_numerical = self.numerical.shape[1]
        self.num_categorical = self.categorical.shape[1]
        self.num_features = self.num_numerical + self.num_categorical
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
                - 'numerical': Tensor[num_numerical]
                - 'categorical': Tensor[num_categorical]
                - 'labels': Tensor (scalar, if include_labels=True)
        """
        item = {
            'numerical': self.numerical[idx],
            'categorical': self.categorical[idx],
        }
        
        if self.include_labels:
            item['labels'] = self.labels[idx]
        
        return item
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        preprocessor: TabularPreprocessor,
        include_labels: bool = True,
    ) -> "TabularDataset":
        """Create dataset directly from DataFrame using fitted preprocessor."""
        data = preprocessor.transform(df, return_labels=include_labels)
        return cls(data, include_labels=include_labels)


class OpenMLDataset:
    """
    Wrapper for loading datasets from OpenML.
    
    Provides easy access to benchmark datasets like Adult, Covertype, etc.
    Handles train/val/test splits and preprocessing.
    
    Example:
        >>> dataset = OpenMLDataset.from_task_id(task_id=7592)  # Adult
        >>> train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=256)
    """
    
    # Common benchmark dataset task IDs
    BENCHMARKS = {
        'adult': 7592,
        'covertype': 7593,
        'higgs': 146606,
        'jannis': 168868,
        'helena': 168329,
    }
    
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "unknown",
        preprocessor: Optional[TabularPreprocessor] = None,
    ):
        """
        Args:
            X: Feature DataFrame
            y: Target Series
            dataset_name: Name for logging
            preprocessor: Optional pre-fitted preprocessor
        """
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor or TabularPreprocessor()
        
        self._is_prepared = False
        self.train_data: Optional[Dict[str, torch.Tensor]] = None
        self.val_data: Optional[Dict[str, torch.Tensor]] = None
        self.test_data: Optional[Dict[str, torch.Tensor]] = None
    
    @classmethod
    def from_task_id(cls, task_id: int) -> "OpenMLDataset":
        """Load dataset from OpenML task ID."""
        task = openml.tasks.get_task(task_id)
        dataset = task.get_dataset()
        X, y, _, _ = dataset.get_data(
            target=dataset.default_target_attribute,
            dataset_format="dataframe"
        )
        return cls(X, y, dataset_name=dataset.name)
    
    @classmethod
    def from_name(cls, name: str) -> "OpenMLDataset":
        """Load dataset by common name (e.g., 'adult', 'covertype')."""
        name_lower = name.lower()
        if name_lower not in cls.BENCHMARKS:
            raise ValueError(
                f"Unknown dataset: {name}. "
                f"Available: {list(cls.BENCHMARKS.keys())}"
            )
        return cls.from_task_id(cls.BENCHMARKS[name_lower])
    
    @classmethod
    def from_openml_suite(
        cls, 
        suite_name: str = 'OpenML-CC18',
        max_datasets: Optional[int] = None,
    ) -> List["OpenMLDataset"]:
        """Load multiple datasets from an OpenML benchmark suite."""
        suite = openml.study.get_study(suite_name, 'tasks')
        task_ids = suite.tasks[:max_datasets] if max_datasets else suite.tasks
        
        datasets = []
        for task_id in task_ids:
            try:
                datasets.append(cls.from_task_id(task_id))
            except Exception as e:
                print(f"Warning: Failed to load task {task_id}: {e}")
        
        return datasets
    
    def prepare(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42,
    ) -> "OpenMLDataset":
        """
        Prepare train/val/test splits and fit preprocessor.
        
        Args:
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            self for chaining
        """
        # Combine X and y for splitting
        df = self.X.copy()
        df['__target__'] = self.y.values
        
        # Shuffle and split
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        n = len(df)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_test - n_val
        
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train:n_train + n_val]
        test_df = df.iloc[n_train + n_val:]
        
        # Fit preprocessor on training data only
        self.preprocessor.fit(train_df, target_col='__target__')
        
        # Transform all splits
        self.train_data = self.preprocessor.transform(train_df, return_labels=True)
        self.val_data = self.preprocessor.transform(val_df, return_labels=True)
        self.test_data = self.preprocessor.transform(test_df, return_labels=True)
        
        self._is_prepared = True
        
        print(f"Dataset: {self.dataset_name}")
        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        print(f"  Features: {self.preprocessor.get_config()['num_features']} "
              f"(numerical={len(self.preprocessor.numerical_cols)}, "
              f"categorical={len(self.preprocessor.categorical_cols)})")
        print(f"  Classes: {self.preprocessor.num_classes}")
        
        return self
    
    def get_datasets(
        self,
        include_labels: bool = True,
    ) -> Tuple[TabularDataset, TabularDataset, TabularDataset]:
        """Get train, val, test PyTorch Datasets."""
        if not self._is_prepared:
            self.prepare()
        
        return (
            TabularDataset(self.train_data, include_labels=include_labels),
            TabularDataset(self.val_data, include_labels=include_labels),
            TabularDataset(self.test_data, include_labels=include_labels),
        )
    
    def get_dataloaders(
        self,
        batch_size: int = 256,
        num_workers: int = 0,
        pin_memory: bool = True,
        include_labels: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get train, val, test DataLoaders.
        
        Args:
            batch_size: Batch size for all loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            include_labels: Whether to include labels
            
        Returns:
            (train_loader, val_loader, test_loader)
        """
        train_ds, val_ds, test_ds = self.get_datasets(include_labels=include_labels)
        
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        
        return train_loader, val_loader, test_loader
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get configuration dict for model initialization."""
        if not self._is_prepared:
            self.prepare()
        return self.preprocessor.get_config()
    
    def __repr__(self) -> str:
        status = "prepared" if self._is_prepared else "not prepared"
        return f"OpenMLDataset(name={self.dataset_name}, samples={len(self.X)}, {status})"


def create_dataloaders(
    dataset_name: str,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 0,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    Convenience function to create dataloaders from dataset name.
    
    Args:
        dataset_name: Name of dataset ('adult', 'covertype', etc.)
        batch_size: Batch size
        val_ratio: Validation split ratio
        test_ratio: Test split ratio
        num_workers: DataLoader workers
        random_state: Random seed
        
    Returns:
        (train_loader, val_loader, test_loader, model_config)
    """
    dataset = OpenMLDataset.from_name(dataset_name)
    dataset.prepare(val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state)
    
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    model_config = dataset.get_model_config()
    
    return train_loader, val_loader, test_loader, model_config
