"""
TabularPreprocessor: Handles preprocessing of mixed-type tabular data.

Converts raw DataFrames into tensors suitable for the Tabular-JEPA model.
- Numerical features: StandardScaler normalization
- Categorical features: LabelEncoder + handles unknown categories
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder


@dataclass
class ColumnInfo:
    """Metadata for a single column."""
    name: str
    dtype: str  # 'numerical' or 'categorical'
    num_categories: Optional[int] = None  # Only for categorical
    idx: int = 0  # Position in feature order


class TabularPreprocessor:
    """
    Handles preprocessing of mixed-type tabular data for Tabular-JEPA.
    
    Features:
        - Automatic detection of numerical vs categorical columns
        - StandardScaler for numerical features
        - LabelEncoder for categorical features with unknown handling
        - Serialization for reproducibility
    
    Example:
        >>> preprocessor = TabularPreprocessor()
        >>> preprocessor.fit(train_df, target_col='label')
        >>> tensors = preprocessor.transform(train_df)
        >>> # tensors = {'numerical': Tensor, 'categorical': Tensor, 'labels': Tensor}
    """
    
    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        auto_detect: bool = True,
        categorical_threshold: int = 20,
    ):
        """
        Args:
            numerical_cols: List of numerical column names (optional if auto_detect=True)
            categorical_cols: List of categorical column names (optional if auto_detect=True)
            auto_detect: Whether to automatically detect column types
            categorical_threshold: Max unique values to consider as categorical when auto-detecting
        """
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.auto_detect = auto_detect
        self.categorical_threshold = categorical_threshold
        
        # Fitted state
        self.column_info: Dict[str, ColumnInfo] = {}
        self.feature_order: List[str] = []  # Order of features in output tensor
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.category_sizes: List[int] = []  # Vocab size for each categorical
        
        # Target handling
        self.target_col: Optional[str] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.num_classes: Optional[int] = None
        
        self._is_fitted = False
    
    def _detect_column_types(self, df: pd.DataFrame, target_col: Optional[str] = None) -> None:
        """Automatically detect numerical vs categorical columns."""
        for col in df.columns:
            if col == target_col:
                continue
                
            if col in self.numerical_cols or col in self.categorical_cols:
                continue  # Already specified
            
            # Auto-detection logic
            if df[col].dtype in ['object', 'category', 'bool']:
                self.categorical_cols.append(col)
            elif df[col].nunique() <= self.categorical_threshold and df[col].dtype in ['int64', 'int32']:
                # Low-cardinality integers treated as categorical
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)
    
    def fit(
        self, 
        df: pd.DataFrame, 
        target_col: Optional[str] = None
    ) -> "TabularPreprocessor":
        """
        Fit the preprocessor on training data.
        
        Args:
            df: Training DataFrame
            target_col: Name of target column (will be encoded separately)
            
        Returns:
            self for chaining
        """
        self.target_col = target_col
        
        # Auto-detect column types if enabled
        if self.auto_detect:
            self._detect_column_types(df, target_col)
        
        # Build feature order: numerical first, then categorical
        self.feature_order = self.numerical_cols + self.categorical_cols
        
        # Fit numerical scalers
        for idx, col in enumerate(self.numerical_cols):
            scaler = StandardScaler()
            # Handle missing values by filling with median before fitting
            values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
            scaler.fit(values)
            self.scalers[col] = scaler
            
            self.column_info[col] = ColumnInfo(
                name=col,
                dtype='numerical',
                idx=idx,
            )
        
        # Fit categorical encoders
        self.category_sizes = []
        for idx, col in enumerate(self.categorical_cols):
            encoder = LabelEncoder()
            # Convert to string and handle missing
            values = df[col].fillna('__MISSING__').astype(str)
            encoder.fit(values)
            self.label_encoders[col] = encoder
            
            num_categories = len(encoder.classes_)
            self.category_sizes.append(num_categories)
            
            self.column_info[col] = ColumnInfo(
                name=col,
                dtype='categorical',
                num_categories=num_categories,
                idx=len(self.numerical_cols) + idx,
            )
        
        # Fit target encoder if classification
        if target_col is not None and target_col in df.columns:
            self.target_encoder = LabelEncoder()
            self.target_encoder.fit(df[target_col].values)
            self.num_classes = len(self.target_encoder.classes_)
        
        self._is_fitted = True
        return self
    
    def transform(
        self, 
        df: pd.DataFrame,
        return_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform DataFrame to tensors.
        
        Args:
            df: DataFrame to transform
            return_labels: Whether to include labels in output
            
        Returns:
            Dictionary with:
                - 'numerical': Tensor[batch, num_numerical] (float32)
                - 'categorical': Tensor[batch, num_categorical] (int64)
                - 'labels': Tensor[batch] (int64, only if return_labels and target exists)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform. Call fit() first.")
        
        result = {}
        batch_size = len(df)
        
        # Transform numerical features
        if self.numerical_cols:
            numerical_data = np.zeros((batch_size, len(self.numerical_cols)), dtype=np.float32)
            for i, col in enumerate(self.numerical_cols):
                values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
                numerical_data[:, i] = self.scalers[col].transform(values).flatten()
            result['numerical'] = torch.tensor(numerical_data, dtype=torch.float32)
        else:
            result['numerical'] = torch.zeros((batch_size, 0), dtype=torch.float32)
        
        # Transform categorical features
        if self.categorical_cols:
            categorical_data = np.zeros((batch_size, len(self.categorical_cols)), dtype=np.int64)
            for i, col in enumerate(self.categorical_cols):
                values = df[col].fillna('__MISSING__').astype(str)
                # Handle unknown categories
                encoded = np.zeros(len(values), dtype=np.int64)
                for j, val in enumerate(values):
                    if val in self.label_encoders[col].classes_:
                        encoded[j] = self.label_encoders[col].transform([val])[0] + 1  # +1 for padding_idx=0
                    else:
                        encoded[j] = 0  # Unknown -> padding index
                categorical_data[:, i] = encoded
            result['categorical'] = torch.tensor(categorical_data, dtype=torch.int64)
        else:
            result['categorical'] = torch.zeros((batch_size, 0), dtype=torch.int64)
        
        # Transform labels
        if return_labels and self.target_col and self.target_col in df.columns:
            labels = self.target_encoder.transform(df[self.target_col].values)
            result['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        return result
    
    def fit_transform(
        self, 
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        return_labels: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Fit and transform in one step."""
        return self.fit(df, target_col).transform(df, return_labels)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration for model initialization.
        
        Returns:
            Dictionary with:
                - num_numerical: Number of numerical features
                - category_sizes: List of vocabulary sizes for each categorical
                - feature_order: List of feature names in order
                - num_classes: Number of target classes (if applicable)
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fitted first.")
        
        return {
            'num_numerical': len(self.numerical_cols),
            'num_categorical': len(self.categorical_cols),
            'category_sizes': self.category_sizes.copy(),
            'feature_order': self.feature_order.copy(),
            'num_features': len(self.feature_order),
            'num_classes': self.num_classes,
            'column_info': {k: vars(v) for k, v in self.column_info.items()},
        }
    
    def save(self, path: str) -> None:
        """Serialize preprocessor state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'feature_order': self.feature_order,
            'scalers': self.scalers,
            'label_encoders': self.label_encoders,
            'category_sizes': self.category_sizes,
            'column_info': self.column_info,
            'target_col': self.target_col,
            'target_encoder': self.target_encoder,
            'num_classes': self.num_classes,
            '_is_fitted': self._is_fitted,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> "TabularPreprocessor":
        """Load preprocessor from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        preprocessor = cls(
            numerical_cols=state['numerical_cols'],
            categorical_cols=state['categorical_cols'],
            auto_detect=False,
        )
        preprocessor.feature_order = state['feature_order']
        preprocessor.scalers = state['scalers']
        preprocessor.label_encoders = state['label_encoders']
        preprocessor.category_sizes = state['category_sizes']
        preprocessor.column_info = state['column_info']
        preprocessor.target_col = state['target_col']
        preprocessor.target_encoder = state['target_encoder']
        preprocessor.num_classes = state['num_classes']
        preprocessor._is_fitted = state['_is_fitted']
        
        return preprocessor
    
    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"TabularPreprocessor({status}, "
            f"num_numerical={len(self.numerical_cols)}, "
            f"num_categorical={len(self.categorical_cols)})"
        )
