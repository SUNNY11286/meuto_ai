"""
Custom exception classes for the anomaly detection system.
Provides structured error handling with detailed context.
"""

from typing import Optional, Dict, Any


class AnomalyDetectionError(Exception):
    """Base exception class for anomaly detection system."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and context.
        
        Args:
            message: Error message
            context: Additional context information
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of the exception."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class DataLoadingError(AnomalyDetectionError):
    """Exception raised when data loading fails."""
    pass


class ModelError(AnomalyDetectionError):
    """Exception raised when model operations fail."""
    pass


class TrainingError(AnomalyDetectionError):
    """Exception raised when training fails."""
    pass


class ValidationError(AnomalyDetectionError):
    """Exception raised when validation fails."""
    pass


class ConfigurationError(AnomalyDetectionError):
    """Exception raised when configuration is invalid."""
    pass


class PreprocessingError(AnomalyDetectionError):
    """Exception raised when preprocessing fails."""
    pass


class PredictionError(AnomalyDetectionError):
    """Exception raised when prediction fails."""
    pass


class VisualizationError(AnomalyDetectionError):
    """Exception raised when visualization fails."""
    pass


def handle_exceptions(exception_type: type = AnomalyDetectionError):
    """
    Decorator to handle exceptions and convert them to custom types.
    
    Args:
        exception_type: Type of exception to convert to
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type:
                # Re-raise custom exceptions as-is
                raise
            except Exception as e:
                # Convert other exceptions to custom type
                raise exception_type(
                    f"Error in {func.__name__}: {str(e)}",
                    context={
                        "function": func.__name__,
                        "args": str(args)[:100],  # Truncate for readability
                        "kwargs": str(kwargs)[:100],
                        "original_exception": type(e).__name__
                    }
                ) from e
        return wrapper
    return decorator
