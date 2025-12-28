"""Losses module for Tabular-JEPA."""

from .jepa_loss import (
    JEPALoss,
    InfoNCELoss,
    create_loss_function,
)

__all__ = [
    'JEPALoss',
    'InfoNCELoss',
    'create_loss_function',
]
