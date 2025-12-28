"""
Unit tests for tokenizer and encoder components.

Run with: python tests/test_tokenizer.py
Or with pytest: python -m pytest tests/test_tokenizer.py -v
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.tokenizer import (
    NumericalEmbedding,
    CategoricalEmbedding,
    PositionalEncoding,
    FeatureTokenizer,
)
from models.encoder import TransformerEncoder, TransformerEncoderBlock
from models.masking import RandomMasking, BlockMasking, MaskingOutput


def test_numerical_embedding():
    """Test NumericalEmbedding forward pass and shapes."""
    print("Testing NumericalEmbedding...")
    
    batch_size = 32
    num_numerical = 5
    d_model = 64
    
    emb = NumericalEmbedding(num_numerical, d_model)
    x = torch.randn(batch_size, num_numerical)
    
    output = emb(x)
    
    assert output.shape == (batch_size, num_numerical, d_model), \
        f"Expected shape {(batch_size, num_numerical, d_model)}, got {output.shape}"
    
    # Test with zero numerical features
    emb_zero = NumericalEmbedding(0, d_model)
    x_zero = torch.randn(batch_size, 0)
    output_zero = emb_zero(x_zero)
    assert output_zero.shape == (batch_size, 0, d_model)
    
    print("✓ NumericalEmbedding tests passed!")


def test_categorical_embedding():
    """Test CategoricalEmbedding forward pass and shapes."""
    print("Testing CategoricalEmbedding...")
    
    batch_size = 32
    category_sizes = [10, 20, 5]  # 3 categorical features
    d_model = 64
    
    emb = CategoricalEmbedding(category_sizes, d_model)
    x = torch.randint(0, 5, (batch_size, len(category_sizes)))
    
    output = emb(x)
    
    assert output.shape == (batch_size, len(category_sizes), d_model), \
        f"Expected shape {(batch_size, len(category_sizes), d_model)}, got {output.shape}"
    
    # Test that padding (index 0) produces zero embedding
    x_padding = torch.zeros(batch_size, len(category_sizes), dtype=torch.long)
    output_padding = emb(x_padding)
    # Padding embeddings should be close to zero (initialized as zero)
    assert torch.allclose(output_padding, torch.zeros_like(output_padding), atol=1e-6), \
        "Padding embeddings should be zero"
    
    # Test with empty categorical
    emb_empty = CategoricalEmbedding([], d_model)
    x_empty = torch.randint(0, 5, (batch_size, 0))
    output_empty = emb_empty(x_empty)
    assert output_empty.shape == (batch_size, 0, d_model)
    
    print("✓ CategoricalEmbedding tests passed!")


def test_positional_encoding():
    """Test PositionalEncoding (learnable and sinusoidal)."""
    print("Testing PositionalEncoding...")
    
    batch_size = 32
    num_positions = 10
    d_model = 64
    
    # Learnable positional encoding
    pe_learnable = PositionalEncoding(num_positions, d_model, learnable=True)
    x = torch.randn(batch_size, num_positions, d_model)
    output = pe_learnable(x)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    # Check that positional encoding is added (output != input)
    assert not torch.allclose(output, x), \
        "Positional encoding should modify the input"
    
    # Sinusoidal positional encoding
    pe_sinusoidal = PositionalEncoding(num_positions, d_model, learnable=False)
    output_sin = pe_sinusoidal(x)
    assert output_sin.shape == x.shape
    
    # Test get_encoding with indices
    indices = torch.randint(0, num_positions, (batch_size, 5))
    encoding = pe_learnable.get_encoding(indices)
    assert encoding.shape == (batch_size, 5, d_model)
    
    print("✓ PositionalEncoding tests passed!")


def test_feature_tokenizer():
    """Test full FeatureTokenizer pipeline."""
    print("Testing FeatureTokenizer...")
    
    batch_size = 32
    num_numerical = 5
    category_sizes = [10, 20, 5]
    d_model = 64
    num_features = num_numerical + len(category_sizes)
    
    tokenizer = FeatureTokenizer(
        num_numerical=num_numerical,
        category_sizes=category_sizes,
        d_model=d_model,
    )
    
    numerical = torch.randn(batch_size, num_numerical)
    categorical = torch.randint(1, 5, (batch_size, len(category_sizes)))
    
    tokens = tokenizer(numerical, categorical)
    
    assert tokens.shape == (batch_size, num_features, d_model), \
        f"Expected shape {(batch_size, num_features, d_model)}, got {tokens.shape}"
    
    # Test with return_before_pos
    tokens_after, tokens_before = tokenizer(numerical, categorical, return_before_pos=True)
    assert tokens_after.shape == tokens_before.shape
    assert not torch.allclose(tokens_after, tokens_before), \
        "Tokens before and after positional encoding should differ"
    
    # Test from_config
    config = {
        'num_numerical': num_numerical,
        'category_sizes': category_sizes,
        'feature_order': [f'f{i}' for i in range(num_features)],
    }
    tokenizer2 = FeatureTokenizer.from_config(config, d_model=d_model)
    tokens2 = tokenizer2(numerical, categorical)
    assert tokens2.shape == tokens.shape
    
    print("✓ FeatureTokenizer tests passed!")


def test_transformer_encoder_block():
    """Test single TransformerEncoderBlock."""
    print("Testing TransformerEncoderBlock...")
    
    batch_size = 32
    seq_len = 10
    d_model = 64
    
    block = TransformerEncoderBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=256,
        dropout=0.0,  # No dropout for deterministic testing
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = block(x)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    # Output should be different from input (not identity)
    assert not torch.allclose(output, x), \
        "Encoder block should transform the input"
    
    print("✓ TransformerEncoderBlock tests passed!")


def test_transformer_encoder():
    """Test full TransformerEncoder."""
    print("Testing TransformerEncoder...")
    
    batch_size = 32
    seq_len = 10
    d_model = 64
    n_layers = 4
    
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=256,
        dropout=0.0,
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = encoder(x)
    
    assert output.shape == x.shape, \
        f"Expected shape {x.shape}, got {output.shape}"
    
    # Test return_all_layers
    all_outputs = encoder(x, return_all_layers=True)
    assert len(all_outputs) == n_layers, \
        f"Expected {n_layers} layer outputs, got {len(all_outputs)}"
    for i, layer_out in enumerate(all_outputs):
        assert layer_out.shape == x.shape, \
            f"Layer {i} output shape mismatch"
    
    # Test gradient flow
    output = encoder(x)
    loss = output.sum()
    loss.backward()
    
    # Check that all parameters have gradients
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
    
    print("✓ TransformerEncoder tests passed!")


def test_random_masking():
    """Test RandomMasking strategy."""
    print("Testing RandomMasking...")
    
    batch_size = 32
    num_features = 10
    d_model = 64
    mask_ratio = 0.5
    
    masking = RandomMasking(mask_ratio=mask_ratio)
    tokens = torch.randn(batch_size, num_features, d_model)
    
    output = masking(tokens)
    
    assert isinstance(output, MaskingOutput)
    
    # Check shapes
    num_masked = int(num_features * mask_ratio)
    num_visible = num_features - num_masked
    
    assert output.visible_indices.shape == (batch_size, num_visible), \
        f"visible_indices shape mismatch: {output.visible_indices.shape}"
    assert output.masked_indices.shape == (batch_size, num_masked), \
        f"masked_indices shape mismatch: {output.masked_indices.shape}"
    assert output.visible_tokens.shape == (batch_size, num_visible, d_model), \
        f"visible_tokens shape mismatch: {output.visible_tokens.shape}"
    assert output.masked_tokens.shape == (batch_size, num_masked, d_model), \
        f"masked_tokens shape mismatch: {output.masked_tokens.shape}"
    assert output.visible_mask.shape == (batch_size, num_features), \
        f"visible_mask shape mismatch: {output.visible_mask.shape}"
    
    # Check that indices are valid
    assert (output.visible_indices >= 0).all() and (output.visible_indices < num_features).all()
    assert (output.masked_indices >= 0).all() and (output.masked_indices < num_features).all()
    
    # Check that visible + masked covers all features
    all_indices = torch.cat([output.visible_indices, output.masked_indices], dim=1)
    for b in range(batch_size):
        unique_indices = torch.unique(all_indices[b])
        assert len(unique_indices) == num_features, \
            f"Batch {b}: Expected {num_features} unique indices, got {len(unique_indices)}"
    
    # Check that tokens match indices
    for b in range(min(3, batch_size)):  # Check first 3 samples
        for i, idx in enumerate(output.visible_indices[b]):
            assert torch.allclose(output.visible_tokens[b, i], tokens[b, idx]), \
                f"Visible token mismatch at batch {b}, position {i}"
    
    print("✓ RandomMasking tests passed!")


def test_block_masking():
    """Test BlockMasking strategy."""
    print("Testing BlockMasking...")
    
    batch_size = 8
    num_features = 12
    d_model = 64
    
    masking = BlockMasking(num_blocks=2, mask_ratio=0.5)
    tokens = torch.randn(batch_size, num_features, d_model)
    
    output = masking(tokens)
    
    assert isinstance(output, MaskingOutput)
    assert output.visible_tokens.shape[0] == batch_size
    assert output.masked_tokens.shape[0] == batch_size
    assert output.visible_tokens.shape[2] == d_model
    assert output.masked_tokens.shape[2] == d_model
    
    # Visible + masked should not exceed total features
    num_visible = output.visible_tokens.shape[1]
    num_masked = output.masked_tokens.shape[1]
    assert num_visible + num_masked <= num_features, \
        f"Total {num_visible + num_masked} exceeds num_features {num_features}"
    
    # Should have at least 1 visible and 1 masked
    assert num_visible >= 1, "Should have at least 1 visible token"
    assert num_masked >= 1, "Should have at least 1 masked token"
    
    print("✓ BlockMasking tests passed!")


def test_end_to_end_tokenizer_encoder():
    """Test full pipeline: Tokenizer → Masking → Encoder."""
    print("Testing end-to-end Tokenizer → Masking → Encoder...")
    
    batch_size = 16
    num_numerical = 5
    category_sizes = [10, 20, 5]
    d_model = 64
    num_features = num_numerical + len(category_sizes)
    
    # Create components
    tokenizer = FeatureTokenizer(
        num_numerical=num_numerical,
        category_sizes=category_sizes,
        d_model=d_model,
    )
    
    masking = RandomMasking(mask_ratio=0.5)
    
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=4,
        n_layers=2,
    )
    
    # Create input
    numerical = torch.randn(batch_size, num_numerical)
    categorical = torch.randint(1, 5, (batch_size, len(category_sizes)))
    
    # Forward pass
    tokens = tokenizer(numerical, categorical)
    assert tokens.shape == (batch_size, num_features, d_model)
    
    mask_output = masking(tokens)
    
    # Encode visible tokens only (as in JEPA context encoder)
    visible_encoded = encoder(mask_output.visible_tokens)
    assert visible_encoded.shape == mask_output.visible_tokens.shape
    
    # Encode all tokens (as in JEPA target encoder)
    full_encoded = encoder(tokens)
    assert full_encoded.shape == tokens.shape
    
    # Test gradient flow
    loss = visible_encoded.sum() + full_encoded.sum()
    loss.backward()
    
    print("✓ End-to-end pipeline tests passed!")


def test_parameter_count():
    """Verify model parameter counts are reasonable."""
    print("Testing parameter counts...")
    
    d_model = 128
    n_layers = 6
    
    encoder = TransformerEncoder(
        d_model=d_model,
        n_heads=4,
        n_layers=n_layers,
        d_ff=512,
    )
    
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    
    print(f"  Encoder: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Rough sanity check: 6-layer transformer with d_model=128 should have ~2-5M params
    assert 100_000 < total_params < 10_000_000, \
        f"Unexpected parameter count: {total_params}"
    
    tokenizer = FeatureTokenizer(
        num_numerical=10,
        category_sizes=[50] * 5,
        d_model=d_model,
    )
    
    tokenizer_params = sum(p.numel() for p in tokenizer.parameters())
    print(f"  Tokenizer: {tokenizer_params:,} params")
    
    print("✓ Parameter count tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Running Tokenizer & Encoder Tests")
    print("=" * 60 + "\n")
    
    test_numerical_embedding()
    test_categorical_embedding()
    test_positional_encoding()
    test_feature_tokenizer()
    test_transformer_encoder_block()
    test_transformer_encoder()
    test_random_masking()
    test_block_masking()
    test_end_to_end_tokenizer_encoder()
    test_parameter_count()
    
    print("\n" + "=" * 60)
    print("All Step 2 tests passed! ✓")
    print("=" * 60)
