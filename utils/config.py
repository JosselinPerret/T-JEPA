"""
Configuration management for Tabular-JEPA.

Uses OmegaConf for hierarchical YAML configuration with:
- Default configs
- Experiment-specific overrides
- Command-line overrides
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict
from pathlib import Path
import os

from omegaconf import OmegaConf, DictConfig, MISSING


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset_name: str = "adult"
    batch_size: int = 256
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    num_workers: int = 0
    random_state: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128
    encoder_layers: int = 6
    encoder_heads: int = 4
    encoder_ff_dim: int = 512
    predictor_layers: int = 2
    predictor_heads: int = 4
    predictor_ff_dim: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    predictor_type: str = "transformer"  # 'transformer' or 'mlp'


@dataclass
class JEPAConfig:
    """JEPA-specific configuration."""
    mask_ratio: float = 0.5
    mask_type: str = "random"  # 'random', 'block', 'structured'
    min_masked: int = 1
    ema_decay_base: float = 0.996
    ema_decay_max: float = 1.0
    ema_warmup_steps: int = 1000
    ema_schedule: str = "cosine"  # 'constant', 'cosine', 'linear'


@dataclass
class LossConfig:
    """Loss function configuration."""
    normalize: bool = True
    loss_type: str = "mse"  # 'mse', 'smooth_l1', 'cosine', 'infonce'
    vic_weight: float = 0.0


@dataclass
class PretrainConfig:
    """Pre-training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Optimizer
    optimizer: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    
    # Scheduler
    scheduler: str = "cosine"  # 'cosine', 'linear', 'step', 'none'
    
    # Logging
    log_interval: int = 100
    save_interval: int = 10


@dataclass 
class LinearProbeConfig:
    """Linear probing configuration."""
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    pooling: str = "mean"  # 'mean', 'first', 'cls'


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    name: str = "tabular-jepa"
    output_dir: str = "./outputs"
    seed: int = 42
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    
    # Logging
    use_wandb: bool = False
    use_tensorboard: bool = True
    wandb_project: str = "tabular-jepa"


@dataclass
class Config:
    """Complete configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    linear_probe: LinearProbeConfig = field(default_factory=LinearProbeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def get_default_config() -> DictConfig:
    """Get default configuration as OmegaConf object."""
    schema = OmegaConf.structured(Config)
    return schema


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to YAML config file (optional)
        overrides: List of CLI overrides like ["pretrain.epochs=50", "model.d_model=64"]
        
    Returns:
        Merged configuration
    """
    # Start with defaults
    config = get_default_config()
    
    # Load from file if provided
    if config_path is not None:
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        file_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, file_config)
    
    # Apply CLI overrides
    if overrides:
        override_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, override_config)
    
    return config


def save_config(config: DictConfig, path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, path)


def config_to_dict(config: DictConfig) -> Dict[str, Any]:
    """Convert OmegaConf to plain dictionary."""
    return OmegaConf.to_container(config, resolve=True)


def get_device(device_str: str = "auto") -> str:
    """Get device string, resolving 'auto' to 'cuda' or 'cpu'."""
    import torch
    
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def print_config(config: DictConfig) -> None:
    """Pretty print configuration."""
    print(OmegaConf.to_yaml(config))


# Register resolvers for dynamic config values
def _register_resolvers():
    """Register OmegaConf resolvers for dynamic values."""
    try:
        # Resolver to get current working directory
        OmegaConf.register_new_resolver("cwd", lambda: os.getcwd())
        
        # Resolver for environment variables
        OmegaConf.register_new_resolver(
            "env", 
            lambda x, default="": os.environ.get(x, default)
        )
    except Exception:
        # Resolvers already registered
        pass

_register_resolvers()
