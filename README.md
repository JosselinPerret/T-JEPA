# Tabular-JEPA

A PyTorch implementation of **Tabular-JEPA** (Joint Embedding Predictive Architecture for Tabular Data) - a self-supervised foundation model for tabular data.

## Overview

Tabular-JEPA adapts the JEPA (Joint Embedding Predictive Architecture) framework to tabular data by:

1. **Feature-as-Token Strategy**: Each column (feature) is treated as a separate token
2. **Self-Supervised Pre-training**: Learn representations by predicting masked feature embeddings
3. **Foundation Model**: Pre-train once, fine-tune on multiple downstream tasks

### Architecture

```
Input Table → Feature Tokenizer → [visible_tokens, masked_tokens]
                                        ↓                ↓
                                 Context Encoder   Target Encoder (EMA)
                                        ↓                ↓
                                    Predictor   ←→   Targets
                                        ↓
                                  L2 Loss in Latent Space
```

**Key Components:**
- **Feature Tokenizer**: Separate embeddings for numerical and categorical features
- **Context Encoder**: Transformer processing visible (unmasked) features
- **Target Encoder**: EMA copy of context encoder (no gradients)
- **Predictor**: Lightweight transformer predicting masked feature representations

## Installation

```bash
# Clone the repository
git clone https://github.com/JosselinPerret/T-JEPA.git
cd T-JEPA

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Pre-training

```bash
# Pre-train on Adult dataset with default config
python scripts/pretrain.py --dataset adult --epochs 100

# Pre-train with custom config
python scripts/pretrain.py --config configs/experiments/adult_pretrain.yaml
```

### 2. Linear Probing Evaluation

```bash
# Evaluate pre-trained model
python scripts/evaluate.py --checkpoint outputs/tabular-jepa/checkpoints/best.pt --dataset adult
```

### 3. Benchmarking

```bash
# Run benchmarks on multiple datasets
python scripts/benchmark.py --datasets adult covertype --epochs 100
```

## Project Structure

```
T-JEPA/
├── configs/                    # Configuration files
│   ├── default.yaml           # Default hyperparameters
│   └── experiments/           # Dataset-specific configs
├── data/                       # Data loading and preprocessing
│   ├── preprocessing.py       # TabularPreprocessor
│   ├── datasets.py            # Dataset classes (OpenML integration)
│   └── download_data.py       # Data download utilities
├── models/                     # Model components
│   ├── tokenizer.py           # Feature tokenization (numerical, categorical)
│   ├── encoder.py             # Transformer encoder
│   ├── predictor.py           # JEPA predictor
│   ├── masking.py             # Masking strategies
│   └── jepa.py                # Complete TabularJEPA model
├── training/                   # Training utilities
│   ├── pretrain.py            # Pre-training loop and Trainer class
│   └── ema.py                 # Exponential moving average updater
├── evaluation/                 # Evaluation utilities
│   └── linear_probe.py        # Linear probing evaluator
├── losses/                     # Loss functions
│   └── jepa_loss.py           # JEPA loss with optional VICReg
├── utils/                      # Utilities
│   ├── config.py              # Configuration management
│   ├── logging.py             # Logging (TensorBoard, W&B)
│   └── checkpointing.py       # Checkpoint management
├── scripts/                    # Executable scripts
│   ├── pretrain.py            # Pre-training entry point
│   ├── evaluate.py            # Evaluation entry point
│   └── benchmark.py           # Benchmark script
├── tests/                      # Unit tests
├── outputs/                    # Training outputs (gitignored)
└── requirements.txt
```

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
# Model architecture
model:
  d_model: 128              # Embedding dimension
  encoder_layers: 6         # Transformer layers
  encoder_heads: 4          # Attention heads
  dropout: 0.1

# JEPA settings
jepa:
  mask_ratio: 0.5           # Fraction of features to mask
  mask_type: "random"       # Masking strategy: random, block
  ema_decay_base: 0.996     # EMA decay for target encoder

# Pre-training
pretrain:
  epochs: 100
  batch_size: 256
  learning_rate: 1e-4
  weight_decay: 0.05
  warmup_epochs: 10
  scheduler: "cosine"
```

## Model Usage

### Python API

```python
from models.jepa import TabularJEPA
from data.preprocessing import TabularPreprocessor
from data.datasets import OpenMLDataset, create_dataloaders

# Load and preprocess data
dataset = OpenMLDataset(dataset_name="adult")
preprocessor = TabularPreprocessor()
X_train, _ = dataset.get_split('train')
preprocessor.fit(X_train)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset, preprocessor, batch_size=256
)

# Create model
model = TabularJEPA(
    num_numerical=preprocessor.num_numerical,
    category_sizes=preprocessor.get_category_sizes(),
    d_model=128,
    encoder_layers=6,
)

# Forward pass (self-supervised)
for batch in train_loader:
    output = model(batch['numerical'], batch['categorical'])
    predictions = output['predictions']  # Predicted masked features
    targets = output['targets']          # Target masked features
    
# Extract representations (for downstream tasks)
representations = model.get_representations(
    numerical, categorical, pooling="mean"
)
```

### Downstream Evaluation

```python
from evaluation.linear_probe import evaluate_linear_probe

# Load pre-trained model
model.load_state_dict(torch.load('checkpoint.pt')['model'])

# Linear probing evaluation
metrics = evaluate_linear_probe(
    encoder=model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_classes=2,
    task_type="classification",
    mode="sklearn",  # or "pytorch"
)

print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
```

## Datasets

The framework supports loading datasets from OpenML:

| Dataset | Task | Features | Samples |
|---------|------|----------|---------|
| adult | Classification | 14 | 48,842 |
| covertype | Classification | 54 | 581,012 |
| california_housing | Regression | 8 | 20,640 |

## Key Features

- **Modular Architecture**: Easy to extend and modify components
- **Mixed Precision Training**: Automatic FP16 for faster training on GPUs
- **Gradient Accumulation**: Support for larger effective batch sizes
- **EMA Scheduling**: Cosine/linear EMA decay schedules
- **Multiple Masking Strategies**: Random, Block, Structured
- **TensorBoard/W&B Integration**: Real-time training monitoring
- **Checkpoint Management**: Automatic best/last checkpoint saving

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_jepa.py -v
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- NumPy
- Pandas
- Scikit-learn
- OpenML
- OmegaConf
- TensorBoard (optional)
- Weights & Biases (optional)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tabular_jepa,
  title={Tabular-JEPA: Joint Embedding Predictive Architecture for Tabular Data},
  year={2024},
  url={https://github.com/JosselinPerret/T-JEPA}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- [V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471)
