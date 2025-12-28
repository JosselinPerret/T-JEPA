"""
Data module for Tabular-JEPA.

Provides preprocessing and dataset utilities for tabular data.
"""

from .preprocessing import TabularPreprocessor, ColumnInfo
from .datasets import (
    TabularDataset,
    OpenMLDataset,
    create_dataloaders,
)

__all__ = [
    'TabularPreprocessor',
    'ColumnInfo',
    'TabularDataset',
    'OpenMLDataset',
    'create_dataloaders',
]
