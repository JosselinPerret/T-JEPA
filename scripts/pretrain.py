"""
Pre-training script for Tabular-JEPA.

Usage:
    python scripts/pretrain.py --config configs/default.yaml
    python scripts/pretrain.py --config configs/experiments/adult_pretrain.yaml
    python scripts/pretrain.py --config configs/default.yaml --dataset adult --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from omegaconf import OmegaConf

from data.preprocessing import TabularPreprocessor
from data.datasets import OpenMLDataset, create_dataloaders
from models.jepa import TabularJEPA
from training.pretrain import create_trainer
from utils.config import load_config, save_config, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-train Tabular-JEPA on tabular datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    
    # Dataset overrides
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["adult", "covertype", "higgs", "california_housing"],
        help="Dataset name (overrides config)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Data directory (overrides config)",
    )
    
    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (overrides config)",
    )
    
    # Model overrides
    parser.add_argument(
        "--embed-dim",
        type=int,
        help="Embedding dimension (overrides config)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer layers (overrides config)",
    )
    
    # Experiment
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "auto"],
        help="Device to train on (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed (overrides config)",
    )
    
    return parser.parse_args()


def apply_overrides(config, args):
    """Apply command line overrides to config."""
    if args.dataset:
        config.data.dataset_name = args.dataset
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.epochs:
        config.pretrain.epochs = args.epochs
    if args.batch_size:
        config.pretrain.batch_size = args.batch_size
    if args.lr:
        config.pretrain.learning_rate = args.lr
    if args.embed_dim:
        config.model.embed_dim = args.embed_dim
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.exp_name:
        config.experiment.name = args.exp_name
    if args.device:
        config.experiment.device = args.device
    if args.seed:
        config.experiment.seed = args.seed
    
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main pre-training function."""
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Setup output directory
    output_dir = Path(args.output_dir) / config.experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    save_config(config, str(output_dir / "config.yaml"))
    
    print("=" * 60)
    print("Tabular-JEPA Pre-training")
    print("=" * 60)
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {get_device(config.experiment.device)}")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = OpenMLDataset(
        dataset_name=config.data.dataset_name,
        cache_dir=config.data.data_dir,
        val_split=config.data.val_split,
        test_split=config.data.test_split,
        seed=config.experiment.seed,
    )
    
    # Create preprocessor
    print("Preprocessing data...")
    preprocessor = TabularPreprocessor(
        numerical_strategy='standardize',
        categorical_strategy='label_encode',
    )
    
    # Fit on training data
    X_train, y_train = dataset.get_split('train')
    preprocessor.fit(X_train)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        dataset=dataset,
        preprocessor=preprocessor,
        batch_size=config.pretrain.batch_size,
        num_workers=config.pretrain.num_workers,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"Numerical features: {preprocessor.num_numerical}")
    print(f"Categorical features: {preprocessor.num_categorical}")
    
    # Get category sizes for embedding
    category_sizes = preprocessor.get_category_sizes()
    
    # Create model
    print("\nCreating model...")
    model = TabularJEPA(
        num_numerical=preprocessor.num_numerical,
        category_sizes=category_sizes,
        d_model=config.model.d_model,
        encoder_layers=config.model.encoder_layers,
        encoder_heads=config.model.encoder_heads,
        encoder_ff_dim=config.model.encoder_ff_dim,
        predictor_layers=config.model.predictor_layers,
        predictor_heads=config.model.predictor_heads,
        predictor_ff_dim=config.model.predictor_ff_dim,
        dropout=config.model.dropout,
        mask_type=config.jepa.mask_type,
        mask_ratio=config.jepa.mask_ratio,
    )
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=str(output_dir),
    )
    
    # Resume if specified
    if args.resume:
        trainer.resume(args.resume)
    
    # Train
    print("\nStarting training...")
    metrics = trainer.train()
    
    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"Final val loss: {metrics['final_val_loss']:.4f}")
    print(f"Best val loss: {metrics['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {output_dir / 'checkpoints'}")
    print(f"Logs saved to: {output_dir / 'logs'}")
    
    # Save preprocessor
    preprocessor.save(str(output_dir / "preprocessor.pkl"))
    print(f"Preprocessor saved to: {output_dir / 'preprocessor.pkl'}")


if __name__ == "__main__":
    main()
