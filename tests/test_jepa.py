"""
Unit tests for JEPA model and components.

Run with: python tests/test_jepa.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.predictor import JEPAPredictor, MLPPredictor
from models.jepa import TabularJEPA
from losses.jepa_loss import JEPALoss, InfoNCELoss, create_loss_function
from training.ema import EMAUpdater


def test_jepa_predictor():
    """Test JEPAPredictor forward pass."""
    print("Testing JEPAPredictor...")
    
    batch_size = 16
    num_visible = 5
    num_masked = 3
    num_features = 8
    d_model = 64
    
    predictor = JEPAPredictor(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    
    # Inputs
    context_output = torch.randn(batch_size, num_visible, d_model)
    masked_indices = torch.randint(0, num_features, (batch_size, num_masked))
    positional_encoding = torch.randn(1, num_features, d_model)
    
    # Forward pass
    predictions = predictor(context_output, masked_indices, positional_encoding)
    
    assert predictions.shape == (batch_size, num_masked, d_model), \
        f"Expected shape {(batch_size, num_masked, d_model)}, got {predictions.shape}"
    
    # Test gradient flow
    loss = predictions.sum()
    loss.backward()
    
    # Check gradients exist
    assert predictor.mask_token.grad is not None, "No gradient for mask_token"
    
    print("✓ JEPAPredictor tests passed!")


def test_mlp_predictor():
    """Test MLPPredictor forward pass."""
    print("Testing MLPPredictor...")
    
    batch_size = 16
    num_visible = 5
    num_masked = 3
    num_features = 8
    d_model = 64
    
    predictor = MLPPredictor(
        d_model=d_model,
        hidden_dim=128,
        n_layers=2,
    )
    
    context_output = torch.randn(batch_size, num_visible, d_model)
    masked_indices = torch.randint(0, num_features, (batch_size, num_masked))
    positional_encoding = torch.randn(1, num_features, d_model)
    
    predictions = predictor(context_output, masked_indices, positional_encoding)
    
    assert predictions.shape == (batch_size, num_masked, d_model)
    
    print("✓ MLPPredictor tests passed!")


def test_tabular_jepa_forward():
    """Test TabularJEPA forward pass."""
    print("Testing TabularJEPA forward pass...")
    
    batch_size = 16
    num_numerical = 5
    category_sizes = [10, 20, 5]
    d_model = 64
    num_features = num_numerical + len(category_sizes)
    
    model = TabularJEPA(
        num_numerical=num_numerical,
        category_sizes=category_sizes,
        d_model=d_model,
        encoder_layers=2,
        encoder_heads=4,
        predictor_layers=1,
        mask_ratio=0.5,
    )
    
    # Create input
    numerical = torch.randn(batch_size, num_numerical)
    categorical = torch.randint(1, 5, (batch_size, len(category_sizes)))
    
    # Forward pass
    output = model(numerical, categorical)
    
    # Check output keys
    assert 'predictions' in output
    assert 'targets' in output
    assert 'masked_indices' in output
    assert 'visible_indices' in output
    
    # Check shapes
    num_masked = output['masked_indices'].shape[1]
    assert output['predictions'].shape == (batch_size, num_masked, d_model), \
        f"predictions shape: {output['predictions'].shape}"
    assert output['targets'].shape == (batch_size, num_masked, d_model), \
        f"targets shape: {output['targets'].shape}"
    
    # Targets should be detached (no gradients)
    assert not output['targets'].requires_grad, "Targets should not require grad"
    
    print("✓ TabularJEPA forward pass tests passed!")


def test_tabular_jepa_ema_update():
    """Test EMA update for target encoder."""
    print("Testing TabularJEPA EMA update...")
    
    model = TabularJEPA(
        num_numerical=5,
        category_sizes=[10, 5],
        d_model=32,
        encoder_layers=1,
        ema_decay=0.5,  # Use 0.5 for easier testing
    )
    
    # Store initial target encoder params
    initial_target_params = [p.clone() for p in model.target_encoder.parameters()]
    
    # Modify context encoder
    with torch.no_grad():
        for param in model.context_encoder.parameters():
            param.add_(1.0)  # Add 1 to all params
    
    # Update target encoder
    model.update_target_encoder()
    
    # Check that target encoder params changed
    for i, (param, initial) in enumerate(zip(model.target_encoder.parameters(), initial_target_params)):
        assert not torch.allclose(param, initial), f"Target param {i} did not change"
    
    # With decay=0.5, new_param should be 0.5 * old + 0.5 * (old + 1) = old + 0.5
    for param, initial in zip(model.target_encoder.parameters(), initial_target_params):
        expected = initial + 0.5
        assert torch.allclose(param, expected, atol=1e-6), "EMA update incorrect"
    
    print("✓ TabularJEPA EMA update tests passed!")


def test_tabular_jepa_representations():
    """Test representation extraction for downstream tasks."""
    print("Testing TabularJEPA representations...")
    
    batch_size = 16
    num_numerical = 5
    category_sizes = [10, 5]
    d_model = 64
    num_features = num_numerical + len(category_sizes)
    
    model = TabularJEPA(
        num_numerical=num_numerical,
        category_sizes=category_sizes,
        d_model=d_model,
        encoder_layers=2,
    )
    
    numerical = torch.randn(batch_size, num_numerical)
    categorical = torch.randint(1, 5, (batch_size, len(category_sizes)))
    
    # Test different pooling methods
    rep_mean = model.get_representations(numerical, categorical, pooling="mean")
    assert rep_mean.shape == (batch_size, d_model), f"Mean pooling shape: {rep_mean.shape}"
    
    rep_first = model.get_representations(numerical, categorical, pooling="first")
    assert rep_first.shape == (batch_size, d_model)
    
    rep_none = model.get_representations(numerical, categorical, pooling="none")
    assert rep_none.shape == (batch_size, num_features, d_model)
    
    print("✓ TabularJEPA representations tests passed!")


def test_tabular_jepa_freeze():
    """Test freezing encoder for linear probing."""
    print("Testing TabularJEPA freeze/unfreeze...")
    
    model = TabularJEPA(
        num_numerical=5,
        category_sizes=[10, 5],
        d_model=32,
        encoder_layers=1,
    )
    
    # Count trainable params before freeze
    trainable_before = model.get_num_parameters(trainable_only=True)
    total = model.get_num_parameters(trainable_only=False)
    
    # Freeze encoder
    model.freeze_encoder()
    
    # Count trainable params after freeze
    trainable_after = model.get_num_parameters(trainable_only=True)
    
    assert trainable_after < trainable_before, \
        f"Trainable params should decrease after freeze: {trainable_after} vs {trainable_before}"
    
    # Unfreeze
    model.unfreeze_encoder()
    trainable_unfrozen = model.get_num_parameters(trainable_only=True)
    
    assert trainable_unfrozen == trainable_before, \
        "Trainable params should be restored after unfreeze"
    
    print("✓ TabularJEPA freeze/unfreeze tests passed!")


def test_jepa_loss():
    """Test JEPA loss computation."""
    print("Testing JEPALoss...")
    
    batch_size = 16
    num_masked = 4
    d_model = 64
    
    # Test MSE loss
    predictions = torch.randn(batch_size, num_masked, d_model, requires_grad=True)
    targets = torch.randn(batch_size, num_masked, d_model)
    
    criterion = JEPALoss(normalize=True, loss_type="mse")
    loss_dict = criterion(predictions, targets)
    
    assert 'loss' in loss_dict
    assert 'reconstruction_loss' in loss_dict
    assert loss_dict['loss'].requires_grad
    
    # Test gradient flow
    loss_dict['loss'].backward()
    assert predictions.grad is not None, "Gradients should flow to predictions"
    
    # Test smooth L1 loss
    predictions_sl1 = torch.randn(batch_size, num_masked, d_model, requires_grad=True)
    criterion_sl1 = JEPALoss(normalize=True, loss_type="smooth_l1")
    loss_dict_sl1 = criterion_sl1(predictions_sl1, targets)
    assert loss_dict_sl1['loss'].item() > 0
    
    # Test cosine loss
    predictions_cos = torch.randn(batch_size, num_masked, d_model, requires_grad=True)
    criterion_cos = JEPALoss(normalize=True, loss_type="cosine")
    loss_dict_cos = criterion_cos(predictions_cos, targets)
    assert loss_dict_cos['loss'].item() > 0
    
    print("✓ JEPALoss tests passed!")


def test_jepa_loss_with_vicreg():
    """Test JEPA loss with VICReg regularization."""
    print("Testing JEPALoss with VICReg...")
    
    batch_size = 32
    num_masked = 4
    d_model = 64
    
    predictions = torch.randn(batch_size, num_masked, d_model)
    targets = torch.randn(batch_size, num_masked, d_model)
    
    criterion = JEPALoss(normalize=True, loss_type="mse", vic_weight=0.1)
    loss_dict = criterion(predictions, targets)
    
    assert 'vic_loss' in loss_dict
    assert loss_dict['vic_loss'].item() > 0, "VICReg loss should be positive"
    
    # Total loss should include VICReg
    assert loss_dict['loss'] > loss_dict['reconstruction_loss']
    
    print("✓ JEPALoss with VICReg tests passed!")


def test_infonce_loss():
    """Test InfoNCE contrastive loss."""
    print("Testing InfoNCELoss...")
    
    batch_size = 16
    num_masked = 4
    d_model = 64
    
    predictions = torch.randn(batch_size, num_masked, d_model)
    targets = predictions.clone()  # Exact match should give high accuracy
    
    criterion = InfoNCELoss(temperature=0.1)
    loss_dict = criterion(predictions, targets)
    
    assert 'loss' in loss_dict
    assert 'accuracy' in loss_dict
    assert loss_dict['accuracy'] > 0.9, "Accuracy should be high for identical inputs"
    
    print("✓ InfoNCELoss tests passed!")


def test_ema_updater():
    """Test EMA updater with different schedules."""
    print("Testing EMAUpdater...")
    
    # Constant schedule
    ema_const = EMAUpdater(base_ema=0.996, schedule="constant")
    assert ema_const.get_ema_decay() == 0.996
    ema_const.step = 1000
    assert ema_const.get_ema_decay() == 0.996
    
    # Cosine schedule
    ema_cos = EMAUpdater(base_ema=0.996, max_ema=1.0, total_steps=100, schedule="cosine")
    decay_start = ema_cos.get_ema_decay()
    ema_cos.step = 50
    decay_mid = ema_cos.get_ema_decay()
    ema_cos.step = 100
    decay_end = ema_cos.get_ema_decay()
    
    assert decay_start == 0.996
    assert decay_mid > decay_start
    assert decay_end == 1.0
    
    # Linear schedule
    ema_lin = EMAUpdater(base_ema=0.9, max_ema=1.0, total_steps=100, schedule="linear")
    ema_lin.step = 50
    assert abs(ema_lin.get_ema_decay() - 0.95) < 0.01
    
    print("✓ EMAUpdater tests passed!")


def test_ema_updater_update():
    """Test EMA updater actual parameter update."""
    print("Testing EMAUpdater parameter update...")
    
    # Create two simple models
    model_online = nn.Linear(10, 10)
    model_target = nn.Linear(10, 10)
    
    # Initialize target with same weights
    model_target.load_state_dict(model_online.state_dict())
    
    # Modify online model
    with torch.no_grad():
        for param in model_online.parameters():
            param.add_(1.0)
    
    # Update with EMA
    ema = EMAUpdater(base_ema=0.5, schedule="constant")
    decay = ema.update(model_online.parameters(), model_target.parameters())
    
    assert decay == 0.5
    assert ema.step == 1
    
    # Check state dict
    state = ema.state_dict()
    assert state['step'] == 1
    
    print("✓ EMAUpdater parameter update tests passed!")


def test_full_training_step():
    """Test a complete training step (forward + loss + backward)."""
    print("Testing full training step...")
    
    batch_size = 16
    num_numerical = 5
    category_sizes = [10, 20]
    d_model = 64
    
    # Create model
    model = TabularJEPA(
        num_numerical=num_numerical,
        category_sizes=category_sizes,
        d_model=d_model,
        encoder_layers=2,
        predictor_layers=1,
        mask_ratio=0.5,
    )
    
    # Create loss
    criterion = JEPALoss(normalize=True, loss_type="mse")
    
    # Create EMA updater
    ema = EMAUpdater(base_ema=0.996, schedule="constant")
    
    # Create optimizer (only for trainable params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create input
    numerical = torch.randn(batch_size, num_numerical)
    categorical = torch.randint(1, 5, (batch_size, len(category_sizes)))
    
    # Forward pass
    output = model(numerical, categorical)
    
    # Compute loss
    loss_dict = criterion(output['predictions'], output['targets'])
    loss = loss_dict['loss']
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients exist for trainable params
    has_grads = False
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            has_grads = True
            break
    assert has_grads, "No gradients computed"
    
    # Optimizer step
    optimizer.step()
    
    # EMA update
    ema.update(
        model.context_encoder.parameters(),
        model.target_encoder.parameters(),
    )
    
    print(f"  Loss: {loss.item():.4f}")
    print("✓ Full training step tests passed!")


def test_parameter_counts():
    """Verify model parameter counts are reasonable."""
    print("Testing parameter counts...")
    
    model = TabularJEPA(
        num_numerical=10,
        category_sizes=[50] * 5,
        d_model=128,
        encoder_layers=6,
        encoder_heads=4,
        predictor_layers=2,
    )
    
    total = model.get_num_parameters(trainable_only=False)
    trainable = model.get_num_parameters(trainable_only=True)
    
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    
    # Target encoder params should not be trainable
    target_params = sum(p.numel() for p in model.target_encoder.parameters())
    print(f"  Target encoder params (frozen): {target_params:,}")
    
    # Trainable should be less than total (target encoder is frozen)
    assert trainable < total, "Trainable should be less than total (target encoder frozen)"
    
    print("✓ Parameter count tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running JEPA Model Tests (Step 3)")
    print("=" * 60 + "\n")
    
    test_jepa_predictor()
    test_mlp_predictor()
    test_tabular_jepa_forward()
    test_tabular_jepa_ema_update()
    test_tabular_jepa_representations()
    test_tabular_jepa_freeze()
    test_jepa_loss()
    test_jepa_loss_with_vicreg()
    test_infonce_loss()
    test_ema_updater()
    test_ema_updater_update()
    test_full_training_step()
    test_parameter_counts()
    
    print("\n" + "=" * 60)
    print("All Step 3 tests passed! ✓")
    print("=" * 60)
