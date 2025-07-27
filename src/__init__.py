"""
Anomaly Detection System for Scratches on Text Images

A comprehensive, production-grade system for detecting scratches on text images
using autoencoder-based anomaly detection with localization capabilities.

Key Features:
- Convolutional autoencoder for reconstruction-based anomaly detection
- Advanced data augmentation and preprocessing
- Comprehensive evaluation metrics (precision, recall, F1, AUC-ROC)
- Anomaly localization with bounding boxes and masks
- Interactive visualization and threshold adjustment
- Professional logging, configuration management, and error handling
- Command-line interface for training, evaluation, and prediction
- Extensible architecture for other surface types

Author: Anomaly Detection Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Anomaly Detection Team"
__email__ = "anomaly-detection@company.com"

# Core imports
from .utils import ConfigManager, LoggerManager, setup_logging
from .data import TextImageDataset, DatasetBuilder
from .models import ConvolutionalAutoencoder, create_autoencoder
from .training import AnomalyDetectionTrainer
from .detection import AnomalyDetector, AnomalyLocalizer, AnomalyEvaluator, AnomalyVisualizer
from .interface import cli

__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__email__',
    
    # Core utilities
    'ConfigManager',
    'LoggerManager',
    'setup_logging',
    
    # Data handling
    'TextImageDataset',
    'DatasetBuilder',
    
    # Models
    'ConvolutionalAutoencoder',
    'create_autoencoder',
    
    # Training
    'AnomalyDetectionTrainer',
    
    # Detection and evaluation
    'AnomalyDetector',
    'AnomalyLocalizer',
    'AnomalyEvaluator',
    'AnomalyVisualizer',
    
    # Interface
    'cli'
]
