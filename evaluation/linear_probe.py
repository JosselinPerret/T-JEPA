"""
Evaluation module for Tabular-JEPA.

Provides:
- Linear probing for classification and regression
- Representation extraction
- Benchmark utilities
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from tqdm import tqdm

from models.jepa import TabularJEPA


class LinearProbe(nn.Module):
    """
    Linear probe head for evaluating learned representations.
    
    Args:
        input_dim: Dimension of input representations
        num_classes: Number of output classes (1 for regression)
        task_type: 'classification' or 'regression'
        pooling: How to pool token representations ('mean', 'first', 'cls')
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        task_type: str = "classification",
        pooling: str = "mean",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.task_type = task_type
        self.pooling = pooling
        
        # Linear head
        if task_type == "classification":
            self.head = nn.Linear(input_dim, num_classes)
        else:
            self.head = nn.Linear(input_dim, 1)
    
    def pool_representations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pool token representations to a single vector.
        
        Args:
            x: Token representations [batch_size, num_tokens, dim]
            
        Returns:
            Pooled representation [batch_size, dim]
        """
        if self.pooling == "mean":
            return x.mean(dim=1)
        elif self.pooling == "first":
            return x[:, 0, :]
        elif self.pooling == "cls":
            # Assumes CLS token is at position 0
            return x[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Token representations [batch_size, num_tokens, dim]
            
        Returns:
            Predictions [batch_size, num_classes] or [batch_size, 1]
        """
        pooled = self.pool_representations(x)
        return self.head(pooled)


class LinearProbeEvaluator:
    """
    Evaluator for linear probing pre-trained encoders.
    
    Supports two modes:
    1. sklearn: Extract features once, train sklearn model
    2. pytorch: Train a linear layer end-to-end
    
    Args:
        encoder: Pre-trained encoder (TabularJEPA)
        device: Device to run on
        mode: 'sklearn' or 'pytorch'
    """
    
    def __init__(
        self,
        encoder: TabularJEPA,
        device: str = "cuda",
        mode: str = "sklearn",
    ):
        self.encoder = encoder.to(device)
        self.encoder.eval()
        self.device = device
        self.mode = mode
        
        # Freeze encoder
        self.encoder.freeze_encoder()
    
    @torch.no_grad()
    def extract_features(
        self,
        dataloader: DataLoader,
        pooling: str = "mean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from pre-trained encoder.
        
        Args:
            dataloader: Data loader
            pooling: Pooling strategy
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        features_list = []
        labels_list = []
        
        for batch in tqdm(dataloader, desc="Extracting features"):
            numerical = batch['numerical'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            
            # Get representations from encoder (use target encoder = use_context_encoder=False)
            reps = self.encoder.get_representations(
                numerical,
                categorical,
                pooling="none",
                use_context_encoder=False,  # Use target encoder for evaluation
            )
            
            # Pool representations
            if pooling == "mean":
                pooled = reps.mean(dim=1)
            elif pooling == "first":
                pooled = reps[:, 0, :]
            else:
                pooled = reps.mean(dim=1)
            
            features_list.append(pooled.cpu().numpy())
            
            if 'label' in batch:
                labels_list.append(batch['label'].numpy())
        
        features = np.concatenate(features_list, axis=0)
        
        if labels_list:
            labels = np.concatenate(labels_list, axis=0)
        else:
            labels = np.zeros(len(features))
        
        return features, labels
    
    def fit_sklearn(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        task_type: str = "classification",
        pooling: str = "mean",
    ) -> Dict[str, float]:
        """
        Fit sklearn model on extracted features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            task_type: 'classification' or 'regression'
            pooling: Pooling strategy
            
        Returns:
            Dictionary of metrics
        """
        # Extract features
        X_train, y_train = self.extract_features(train_loader, pooling)
        
        if val_loader is not None:
            X_val, y_val = self.extract_features(val_loader, pooling)
        else:
            X_val, y_val = None, None
        
        # Fit model
        if task_type == "classification":
            model = LogisticRegression(max_iter=1000, n_jobs=-1)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            train_probs = model.predict_proba(X_train)
            
            metrics = {
                'train_accuracy': accuracy_score(y_train, train_preds),
                'train_f1': f1_score(y_train, train_preds, average='weighted'),
            }
            
            # AUC for binary classification
            if len(np.unique(y_train)) == 2:
                metrics['train_auc'] = roc_auc_score(y_train, train_probs[:, 1])
            
            if X_val is not None:
                val_preds = model.predict(X_val)
                val_probs = model.predict_proba(X_val)
                
                metrics['val_accuracy'] = accuracy_score(y_val, val_preds)
                metrics['val_f1'] = f1_score(y_val, val_preds, average='weighted')
                
                if len(np.unique(y_val)) == 2:
                    metrics['val_auc'] = roc_auc_score(y_val, val_probs[:, 1])
        
        else:  # regression
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            
            train_preds = model.predict(X_train)
            
            metrics = {
                'train_mse': mean_squared_error(y_train, train_preds),
                'train_r2': r2_score(y_train, train_preds),
            }
            
            if X_val is not None:
                val_preds = model.predict(X_val)
                metrics['val_mse'] = mean_squared_error(y_val, val_preds)
                metrics['val_r2'] = r2_score(y_val, val_preds)
        
        self.sklearn_model = model
        return metrics
    
    def fit_pytorch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_classes: int,
        task_type: str = "classification",
        epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        pooling: str = "mean",
    ) -> Dict[str, float]:
        """
        Fit PyTorch linear probe on extracted features.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_classes: Number of classes
            task_type: 'classification' or 'regression'
            epochs: Number of training epochs
            learning_rate: Learning rate
            weight_decay: Weight decay
            pooling: Pooling strategy
            
        Returns:
            Dictionary of metrics
        """
        # Create linear probe
        probe = LinearProbe(
            input_dim=self.encoder.d_model,
            num_classes=num_classes,
            task_type=task_type,
            pooling=pooling,
        ).to(self.device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        if task_type == "classification":
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(1, epochs + 1):
            probe.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                numerical = batch['numerical'].to(self.device)
                categorical = batch['categorical'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get representations
                with torch.no_grad():
                    reps = self.encoder.get_representations(
                        numerical,
                        categorical,
                        pooling="none",
                        use_context_encoder=False,  # Use target encoder
                    )
                
                # Forward through probe
                optimizer.zero_grad()
                logits = probe(reps)
                
                if task_type == "classification":
                    loss = criterion(logits, labels.long())
                    preds = logits.argmax(dim=1)
                    train_correct += (preds == labels).sum().item()
                else:
                    loss = criterion(logits.squeeze(), labels.float())
                
                train_total += len(labels)
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            # Validation
            if val_loader is not None:
                val_metrics = self._evaluate_probe(probe, val_loader, task_type)
                
                if task_type == "classification":
                    val_loss = 1 - val_metrics.get('accuracy', 0)
                else:
                    val_loss = val_metrics.get('mse', float('inf'))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_metrics = {f'val_{k}': v for k, v in val_metrics.items()}
        
        # Final train metrics
        train_metrics = self._evaluate_probe(probe, train_loader, task_type)
        
        all_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
        all_metrics.update(best_metrics)
        
        self.probe = probe
        return all_metrics
    
    @torch.no_grad()
    def _evaluate_probe(
        self,
        probe: LinearProbe,
        dataloader: DataLoader,
        task_type: str,
    ) -> Dict[str, float]:
        """Evaluate probe on a dataloader."""
        probe.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        for batch in dataloader:
            numerical = batch['numerical'].to(self.device)
            categorical = batch['categorical'].to(self.device)
            labels = batch['label']
            
            reps = self.encoder.get_representations(
                numerical,
                categorical,
                pooling="none",
                use_context_encoder=False,  # Use target encoder
            )
            logits = probe(reps)
            
            if task_type == "classification":
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_probs.append(probs.cpu().numpy())
            else:
                preds = logits.squeeze()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
        
        preds = np.concatenate(all_preds)
        labels = np.concatenate(all_labels)
        
        if task_type == "classification":
            probs = np.concatenate(all_probs)
            metrics = {
                'accuracy': accuracy_score(labels, preds),
                'f1': f1_score(labels, preds, average='weighted'),
            }
            if len(np.unique(labels)) == 2:
                metrics['auc'] = roc_auc_score(labels, probs[:, 1])
        else:
            metrics = {
                'mse': mean_squared_error(labels, preds),
                'r2': r2_score(labels, preds),
            }
        
        return metrics


def evaluate_linear_probe(
    encoder: TabularJEPA,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    test_loader: Optional[DataLoader],
    num_classes: int,
    task_type: str = "classification",
    mode: str = "sklearn",
    device: str = "cuda",
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function for linear probing evaluation.
    
    Args:
        encoder: Pre-trained encoder
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_classes: Number of classes
        task_type: 'classification' or 'regression'
        mode: 'sklearn' or 'pytorch'
        device: Device to run on
        **kwargs: Additional arguments for fit methods
        
    Returns:
        Dictionary of evaluation metrics
    """
    evaluator = LinearProbeEvaluator(encoder, device=device, mode=mode)
    
    if mode == "sklearn":
        metrics = evaluator.fit_sklearn(
            train_loader,
            val_loader,
            task_type=task_type,
            **kwargs,
        )
        
        # Test set evaluation
        if test_loader is not None:
            X_test, y_test = evaluator.extract_features(test_loader)
            
            if task_type == "classification":
                test_preds = evaluator.sklearn_model.predict(X_test)
                test_probs = evaluator.sklearn_model.predict_proba(X_test)
                
                metrics['test_accuracy'] = accuracy_score(y_test, test_preds)
                metrics['test_f1'] = f1_score(y_test, test_preds, average='weighted')
                
                if len(np.unique(y_test)) == 2:
                    metrics['test_auc'] = roc_auc_score(y_test, test_probs[:, 1])
            else:
                test_preds = evaluator.sklearn_model.predict(X_test)
                metrics['test_mse'] = mean_squared_error(y_test, test_preds)
                metrics['test_r2'] = r2_score(y_test, test_preds)
    
    else:
        metrics = evaluator.fit_pytorch(
            train_loader,
            val_loader,
            num_classes=num_classes,
            task_type=task_type,
            **kwargs,
        )
    
    return metrics
