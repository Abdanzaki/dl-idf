"""
DL-IDF: Deep Learning-Based Intrusion Detection Framework for Industrial IoT
Phase 4 Models Package.
Provides Multi-Layer Perceptron (MLP) and 1D Convolutional Neural Network (1D-CNN) architectures.
"""

from src.models.mlp import build_mlp
from src.models.cnn import build_cnn1d

__all__ = ["build_mlp", "build_cnn1d"]
