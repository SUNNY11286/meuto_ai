"""
Models package for the anomaly detection system.
Provides neural network architectures for anomaly detection.
"""

from .autoencoder import ConvolutionalAutoencoder, create_autoencoder

__all__ = [
    'ConvolutionalAutoencoder',
    'create_autoencoder'
]
