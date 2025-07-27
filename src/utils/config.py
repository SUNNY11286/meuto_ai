"""
Configuration management utilities with validation and type safety.
Implements the Singleton pattern for global configuration access.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass, field
from .logger import LoggerMixin


@dataclass
class DataConfig:
    """Data configuration parameters."""
    dataset_path: str = "anomaly_detection_test_data"
    good_folder: str = "good"
    bad_folder: str = "bad"
    masks_folder: str = "masks"
    test_split: float = 0.1
    val_split: float = 0.1
    image_size: list = field(default_factory=lambda: [256, 256])
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    architecture: str = "autoencoder"
    encoder_channels: list = field(default_factory=lambda: [3, 64, 128, 256, 512])
    decoder_channels: list = field(default_factory=lambda: [512, 256, 128, 64, 3])
    latent_dim: int = 512
    dropout: float = 0.1
    batch_norm: bool = True


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    scheduler: str = "cosine"
    early_stopping_patience: int = 15
    save_best_only: bool = True
    gradient_clip: float = 1.0


@dataclass
class AnomalyDetectionConfig:
    """Anomaly detection configuration parameters."""
    threshold_percentile: float = 95
    min_anomaly_area: int = 100
    morphology_kernel_size: int = 5
    gaussian_blur_sigma: float = 1.0


@dataclass
class AugmentationConfig:
    """Data augmentation configuration parameters."""
    enabled: bool = True
    rotation_limit: int = 10
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    noise_limit: float = 0.1


class ConfigManager(LoggerMixin):
    """
    Singleton configuration manager with validation and type safety.
    Provides centralized access to all configuration parameters.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: Optional[Union[str, Path]] = None):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        if self._config is None:
            self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[Union[str, Path]]):
        """Load and validate configuration."""
        try:
            if config_path is None:
                config_path = self._find_config_file()
            
            if config_path and Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_dict = yaml.safe_load(f)
                self._config = OmegaConf.create(config_dict)
                self.logger.info(f"Loaded configuration from {config_path}")
            else:
                self._config = self._get_default_config()
                self.logger.warning("Using default configuration")
            
            self._validate_config()
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self._config = self._get_default_config()
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in common locations."""
        search_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path("../config/config.yaml"),
            Path("../../config/config.yaml")
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _get_default_config(self) -> DictConfig:
        """Get default configuration."""
        default_config = {
            "project": {
                "name": "text_scratch_detection",
                "version": "1.0.0",
                "description": "Production-grade anomaly detection for scratches on text images"
            },
            "data": {
                "dataset_path": "anomaly_detection_test_data",
                "good_folder": "good",
                "bad_folder": "bad",
                "masks_folder": "masks",
                "test_split": 0.1,
                "val_split": 0.1,
                "image_size": [256, 256],
                "batch_size": 32,
                "num_workers": 4
            },
            "model": {
                "architecture": "autoencoder",
                "encoder_channels": [3, 64, 128, 256, 512],
                "decoder_channels": [512, 256, 128, 64, 3],
                "latent_dim": 512,
                "dropout": 0.1,
                "batch_norm": True
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "weight_decay": 1e-5,
                "scheduler": "cosine",
                "early_stopping_patience": 15,
                "save_best_only": True,
                "gradient_clip": 1.0
            },
            "anomaly_detection": {
                "threshold_percentile": 95,
                "min_anomaly_area": 100,
                "morphology_kernel_size": 5,
                "gaussian_blur_sigma": 1.0
            },
            "augmentation": {
                "enabled": True,
                "rotation_limit": 10,
                "brightness_limit": 0.2,
                "contrast_limit": 0.2,
                "noise_limit": 0.1
            },
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "tensorboard_dir": "runs"
            },
            "paths": {
                "models_dir": "models",
                "results_dir": "results",
                "visualizations_dir": "visualizations"
            }
        }
        
        return OmegaConf.create(default_config)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        try:
            # Validate data configuration
            assert 0 < self._config.data.test_split < 1, "test_split must be between 0 and 1"
            assert 0 < self._config.data.val_split < 1, "val_split must be between 0 and 1"
            assert self._config.data.batch_size > 0, "batch_size must be positive"
            assert len(self._config.data.image_size) == 2, "image_size must be [height, width]"
            
            # Validate training configuration
            assert self._config.training.epochs > 0, "epochs must be positive"
            assert self._config.training.learning_rate > 0, "learning_rate must be positive"
            assert self._config.training.early_stopping_patience > 0, "early_stopping_patience must be positive"
            
            # Validate model configuration
            assert self._config.model.latent_dim > 0, "latent_dim must be positive"
            assert 0 <= self._config.model.dropout <= 1, "dropout must be between 0 and 1"
            
            # Validate anomaly detection configuration
            assert 0 < self._config.anomaly_detection.threshold_percentile <= 100, "threshold_percentile must be between 0 and 100"
            assert self._config.anomaly_detection.min_anomaly_area >= 0, "min_anomaly_area must be non-negative"
            
            self.logger.info("Configuration validation passed")
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}")
    
    @property
    def config(self) -> DictConfig:
        """Get configuration object."""
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        try:
            return OmegaConf.select(self._config, key, default=default)
        except Exception:
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value by key."""
        OmegaConf.set(self._config, key, value)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to file."""
        with open(path, 'w') as f:
            OmegaConf.save(self._config, f)
        self.logger.info(f"Configuration saved to {path}")
    
    def get_data_config(self) -> DataConfig:
        """Get typed data configuration."""
        return DataConfig(**self._config.data)
    
    def get_model_config(self) -> ModelConfig:
        """Get typed model configuration."""
        return ModelConfig(**self._config.model)
    
    def get_training_config(self) -> TrainingConfig:
        """Get typed training configuration."""
        return TrainingConfig(**self._config.training)
    
    def get_anomaly_config(self) -> AnomalyDetectionConfig:
        """Get typed anomaly detection configuration."""
        return AnomalyDetectionConfig(**self._config.anomaly_detection)
    
    def get_augmentation_config(self) -> AugmentationConfig:
        """Get typed augmentation configuration."""
        return AugmentationConfig(**self._config.augmentation)


# Global configuration instance
def get_config(config_path: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Get global configuration instance."""
    return ConfigManager(config_path)


def create_directories(config: ConfigManager):
    """Create necessary directories based on configuration."""
    directories = [
        config.get("logging.log_dir", "logs"),
        config.get("logging.tensorboard_dir", "runs"),
        config.get("paths.models_dir", "models"),
        config.get("paths.results_dir", "results"),
        config.get("paths.visualizations_dir", "visualizations")
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
