"""CPAF-X reusable research components. All transaction-like data is synthetic."""

from .core import (
    generate_synthetic, validate_dataset, generalize, suppress_k,
    privacy_diagnostics, microaggregate,
)

__all__ = ["generate_synthetic", "validate_dataset", "generalize", "suppress_k",
           "privacy_diagnostics", "microaggregate"]
