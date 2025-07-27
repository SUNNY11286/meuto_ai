"""
Professional logging utilities for the anomaly detection system.
Implements structured logging with multiple handlers and formatters.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union
from loguru import logger
import yaml


class LoggerManager:
    """
    Centralized logger management with configurable handlers and formatters.
    Follows enterprise logging best practices.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, level: Optional[str] = None):
        """
        Initialize logger manager with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        if level:
            self.config['level'] = level
        self._setup_logger()
    
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            "level": "INFO",
            "log_dir": "logs",
            "format": "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            "rotation": "10 MB",
            "retention": "30 days",
            "compression": "zip"
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    return {**default_config, **config.get('logging', {})}
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _setup_logger(self):
        """Setup logger with multiple handlers."""
        # Remove default handler
        logger.remove()
        
        # Console handler
        logger.add(
            sys.stdout,
            level=self.config["level"],
            format=self.config["format"],
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File handler
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(exist_ok=True)
        
        logger.add(
            log_dir / "app.log",
            level=self.config["level"],
            format=self.config["format"],
            rotation=self.config["rotation"],
            retention=self.config["retention"],
            compression=self.config["compression"],
            backtrace=True,
            diagnose=True
        )
        
        # Error file handler
        logger.add(
            log_dir / "errors.log",
            level="ERROR",
            format=self.config["format"],
            rotation=self.config["rotation"],
            retention=self.config["retention"],
            compression=self.config["compression"],
            backtrace=True,
            diagnose=True
        )
    
    def get_logger(self, name: str = __name__):
        """Get logger instance with specified name."""
        return logger.bind(name=name)


# Global logger instance
_logger_manager = LoggerManager()
get_logger = _logger_manager.get_logger


def setup_logging(config_path: Optional[Union[str, Path]] = None, level: Optional[str] = None):
    """Setup logging configuration."""
    global _logger_manager
    _logger_manager = LoggerManager(config_path, level=level)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self):
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__)


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger = get_logger(func.__module__)
        logger.info(f"Starting {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed {func.__name__} in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed {func.__name__} after {execution_time:.2f}s: {e}")
            raise
    
    return wrapper


def log_method_calls(cls):
    """Class decorator to log all method calls."""
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            setattr(cls, attr_name, log_execution_time(attr))
    return cls
