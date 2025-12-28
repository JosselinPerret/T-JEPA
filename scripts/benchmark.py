"""
Benchmark script for running experiments on multiple datasets.

Usage:
    python scripts/benchmark.py --datasets adult covertype --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import json
from datetime import datetime

from data.preprocessing import TabularPreprocessor
from data.datasets import OpenMLDataset, create_dataloaders
from models.jepa import TabularJEPA
from training.pretrain import create_trainer
from evaluation.linear_probe import evaluate_linear_probe
from utils.config import load_config, save_config, get_device


# Dataset configurations
DATASETS = {
    "adult": {
        "name": "adult",
        "task": "classification",
        "num_classes": 2,
    },
    "covertype": {
        "name": "covertype",
        "task": "classification",
        "num_classes": 7,
    },
    "california_housing": {
        "name": "california_housing",
        "task": "regression",
        "num_classes": 1,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Tabular-JEPA benchmark experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult"],
        choices=list(DATASETS.keys()),
        help="Datasets to benchmark on",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Base config file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of pre-training epochs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/benchmark",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--skip-pretrain",
        action="store_true",
        help="Skip pre-training (load existing checkpoints)",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    dataset_name: str,
    config,
    output_dir: Path,
    device: str,
    skip_pretrain: bool = False,
) -> dict:
    """
    Run a single benchmark experiment.
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration
        output_dir: Output directory
        device: Device to use
        skip_pretrain: Whether to skip pre-training
        
    Returns:
        Dictionary of results
    """
    dataset_config = DATASETS[dataset_name]
    experiment_dir = output_dir / dataset_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = OpenMLDataset(
        dataset_name=dataset_name,
        cache_dir="./data",
    )
    
    # Create preprocessor
    print("Creating preprocessor...")
    preprocessor = TabularPreprocessor()
    X_train, _ = dataset.get_split('train')
    preprocessor.fit(X_train)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset=dataset,
        preprocessor=preprocessor,
        batch_size=config.pretrain.batch_size,
        num_workers=0,
    )
    
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset) if val_loader else 0}, Test: {len(test_loader.dataset) if test_loader else 0}")
    print(f"Numerical: {preprocessor.num_numerical}, Categorical: {preprocessor.num_categorical}")
    
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
    
    print(f"Parameters: {model.get_num_parameters():,}")
    
    # Pre-training
    pretrain_metrics = {}
    checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
    
    if not skip_pretrain or not checkpoint_path.exists():
        print("\nPre-training...")
        trainer = create_trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=str(experiment_dir),
        )
        
        pretrain_metrics = trainer.train()
        trainer.logger.close()
        
        # Save preprocessor
        preprocessor.save(str(experiment_dir / "preprocessor.pkl"))
    else:
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    
    # Linear probing evaluation
    print("\nRunning linear probing evaluation...")
    probe_metrics = evaluate_linear_probe(
        encoder=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=dataset_config['num_classes'],
        task_type=dataset_config['task'],
        mode="sklearn",
        device=device,
        pooling="mean",
    )
    
    # Compile results
    results = {
        'dataset': dataset_name,
        'task_type': dataset_config['task'],
        'num_classes': dataset_config['num_classes'],
        'num_samples': len(train_loader.dataset),
        'num_features': preprocessor.num_numerical + preprocessor.num_categorical,
        'num_parameters': model.get_num_parameters(),
        'pretrain_metrics': pretrain_metrics,
        'probe_metrics': probe_metrics,
    }
    
    # Print results
    print(f"\nResults for {dataset_name}:")
    if dataset_config['task'] == 'classification':
        print(f"  Test Accuracy: {probe_metrics.get('test_accuracy', 'N/A'):.4f}")
        print(f"  Test F1: {probe_metrics.get('test_f1', 'N/A'):.4f}")
    else:
        print(f"  Test MSE: {probe_metrics.get('test_mse', 'N/A'):.4f}")
        print(f"  Test R2: {probe_metrics.get('test_r2', 'N/A'):.4f}")
    
    # Save results
    with open(experiment_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    config.pretrain.epochs = args.epochs
    config.experiment.seed = args.seed
    
    device = get_device(args.device)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Tabular-JEPA Benchmark")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    
    # Save config
    save_config(config, str(output_dir / "config.yaml"))
    
    # Run experiments
    all_results = {}
    
    for dataset_name in args.datasets:
        try:
            results = run_experiment(
                dataset_name=dataset_name,
                config=config,
                output_dir=output_dir,
                device=device,
                skip_pretrain=args.skip_pretrain,
            )
            all_results[dataset_name] = results
        except Exception as e:
            print(f"Error on {dataset_name}: {e}")
            all_results[dataset_name] = {'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    
    for dataset_name, results in all_results.items():
        if 'error' in results:
            print(f"{dataset_name}: ERROR - {results['error']}")
        else:
            if results['task_type'] == 'classification':
                metric = results['probe_metrics'].get('test_accuracy', 'N/A')
                print(f"{dataset_name}: Accuracy = {metric:.4f}")
            else:
                metric = results['probe_metrics'].get('test_r2', 'N/A')
                print(f"{dataset_name}: R2 = {metric:.4f}")
    
    # Save summary
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
