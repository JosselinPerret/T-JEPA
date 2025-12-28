"""
Models module for Tabular-JEPA.

Contains all neural network components:
- Tokenizer: Feature tokenization (numerical + categorical)
- Encoder: Transformer encoder
- Masking: Feature masking strategies
- Predictor: JEPA predictor
- JEPA: Complete model assembly
"""

from .tokenizer import (
    NumericalEmbedding,
    CategoricalEmbedding,
    PositionalEncoding,
    FeatureTokenizer,
)

from .encoder import (
    TransformerEncoderBlock,
    TransformerEncoder,
)

from .masking import (
    MaskingOutput,
    RandomMasking,
    BlockMasking,
    StructuredMasking,
    create_masking_strategy,
)

from .predictor import (
    JEPAPredictor,
    MLPPredictor,
)

from .jepa import TabularJEPA

__all__ = [
    # Tokenizer
    'NumericalEmbedding',
    'CategoricalEmbedding',
    'PositionalEncoding',
    'FeatureTokenizer',
    # Encoder
    'TransformerEncoderBlock',
    'TransformerEncoder',
    # Masking
    'MaskingOutput',
    'RandomMasking',
    'BlockMasking',
    'StructuredMasking',
    'create_masking_strategy',
    # Predictor
    'JEPAPredictor',
    'MLPPredictor',
    # JEPA
    'TabularJEPA',
]
