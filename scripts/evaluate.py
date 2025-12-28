"""
Linear probing evaluation script for Tabular-JEPA.

Usage:
    python scripts/evaluate.py --checkpoint outputs/tabular-jepa/checkpoints/best.pt --dataset adult
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json

from data.preprocessing import TabularPreprocessor
from data.datasets import OpenMLDataset, create_dataloaders
from models.jepa import TabularJEPA
from evaluation.linear_probe import evaluate_linear_probe
from utils.config import load_config, get_device
from utils.checkpointing import load_model_only


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Tabular-JEPA with linear probing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file (uses checkpoint config if not provided)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="adult",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output file for results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sklearn", "pytorch"],
        default="sklearn",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        choices=["mean", "first"],
        default="mean",
        help="Pooling strategy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on",
    )
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    checkpoint_dir = checkpoint_path.parent.parent
    
    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            config = load_config(str(config_path))
        else:
            config = load_config("configs/default.yaml")
    
    device = get_device(args.device)
    
    print("=" * 60)
    print("Tabular-JEPA Linear Probing Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load preprocessor
    preprocessor_path = checkpoint_dir / "preprocessor.pkl"
    if preprocessor_path.exists():
        print(f"\nLoading preprocessor from {preprocessor_path}")
        preprocessor = TabularPreprocessor.load(str(preprocessor_path))
    else:
        # Create new preprocessor
        print("\nCreating new preprocessor...")
        dataset = OpenMLDataset(
            dataset_name=args.dataset,
            cache_dir=config.data.data_dir if hasattr(config.data, 'data_dir') else './data',
        )
        preprocessor = TabularPreprocessor()
        X_train, _ = dataset.get_split('train')
        preprocessor.fit(X_train)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = OpenMLDataset(
        dataset_name=args.dataset,
        cache_dir=config.data.data_dir if hasattr(config.data, 'data_dir') else './data',
    )
    
    # Create dataloaders with labels
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        preprocessor=preprocessor,
        batch_size=256,
        num_workers=0,
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset) if val_loader else 0}")
    print(f"Test samples: {len(test_loader.dataset) if test_loader else 0}")
    
    # Determine task type and number of classes
    _, y_train = dataset.get_split('train')
    unique_values = y_train.nunique() if hasattr(y_train, 'nunique') else len(set(y_train))
    
    if unique_values <= 20:
        task_type = "classification"
        num_classes = unique_values
        print(f"Task: Classification ({num_classes} classes)")
    else:
        task_type = "regression"
        num_classes = 1
        print(f"Task: Regression")
    
    # Create model
    print("\nCreating model...")
    category_sizes = preprocessor.get_category_sizes()
    
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
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {checkpoint_path}")
    load_model_only(str(checkpoint_path), model, map_location=device)
    
    # Run evaluation
    print("\nRunning linear probing evaluation...")
    metrics = evaluate_linear_probe(
        encoder=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        task_type=task_type,
        mode=args.mode,
        device=device,
        pooling=args.pooling,
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    
    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'dataset': args.dataset,
        'mode': args.mode,
        'task_type': task_type,
        'num_classes': num_classes,
        'metrics': metrics,
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
