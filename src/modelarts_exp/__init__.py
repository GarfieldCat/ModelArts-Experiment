"""
ModelArts Experiment Package

This package contains implementations of machine learning algorithms
for the ModelArts platform experiments.
"""

__version__ = "0.1.0"
__author__ = "Will Mao"

from .gradient_descent import (
    generate_gradient,
    gradient_descending,
    calculate_mse,
    run_demo
)

__all__ = [
    "generate_gradient",
    "gradient_descending",
    "calculate_mse",
    "run_demo"
]
