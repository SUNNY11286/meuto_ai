"""
Anomaly detection module for identifying scratches on text images.
Implements reconstruction-based anomaly detection with localization capabilities.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from scipy import ndimage
from skimage import measure, morphology
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ..utils import LoggerMixin, PredictionError, handle_exceptions
from ..models import ConvolutionalAutoencoder


class AnomalyLocalizer(LoggerMixin):
    """
    Localizes anomalies in images using reconstruction error maps.
    Provides bounding boxes and masks for detected anomalies.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 95,
        min_anomaly_area: int = 100,
        morphology_kernel_size: int = 5,
        gaussian_blur_sigma: float = 1.0
    ):
        """
        Initialize anomaly localizer.
        
        Args:
            threshold_percentile: Percentile for anomaly threshold
            min_anomaly_area: Minimum area for anomaly regions
            morphology_kernel_size: Kernel size for morphological operations
            gaussian_blur_sigma: Sigma for Gaussian blur
        """
        self.threshold_percentile = threshold_percentile
        self.min_anomaly_area = min_anomaly_area
        self.morphology_kernel_size = morphology_kernel_size
        self.gaussian_blur_sigma = gaussian_blur_sigma
        
        self.logger.info("Initialized anomaly localizer")
    
    @handle_exceptions(PredictionError)
    def compute_reconstruction_error(
        self, 
        original: np.ndarray, 
        reconstruction: np.ndarray
    ) -> np.ndarray:
        """
        Compute pixel-wise reconstruction error.
        
        Args:
            original: Original image [H, W, C]
            reconstruction: Reconstructed image [H, W, C]
            
        Returns:
            Error map [H, W]
        """
        # Ensure inputs are numpy arrays
        if isinstance(original, torch.Tensor):
            original = original.cpu().numpy()
        if isinstance(reconstruction, torch.Tensor):
            reconstruction = reconstruction.cpu().numpy()
        
        # Compute squared error
        error = np.mean((original - reconstruction) ** 2, axis=-1)
        
        # Apply Gaussian blur to smooth error map
        if self.gaussian_blur_sigma > 0:
            error = ndimage.gaussian_filter(error, sigma=self.gaussian_blur_sigma)
        
        return error
    
    @handle_exceptions(PredictionError)
    def threshold_error_map(self, error_map: np.ndarray) -> np.ndarray:
        """
        Threshold error map to create binary anomaly mask.
        
        Args:
            error_map: Reconstruction error map
            
        Returns:
            Binary anomaly mask
        """
        # Calculate threshold based on percentile
        threshold = np.percentile(error_map, self.threshold_percentile)
        
        # Create binary mask
        binary_mask = (error_map > threshold).astype(np.uint8)
        
        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        
        # Close small gaps
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        return binary_mask
    
    @handle_exceptions(PredictionError)
    def extract_anomaly_regions(self, binary_mask: np.ndarray) -> List[Dict]:
        """
        Extract individual anomaly regions from binary mask.
        
        Args:
            binary_mask: Binary anomaly mask
            
        Returns:
            List of anomaly region dictionaries
        """
        # Label connected components
        labeled_mask = measure.label(binary_mask)
        regions = measure.regionprops(labeled_mask)
        
        anomaly_regions = []
        
        for region in regions:
            # Filter by area
            if region.area < self.min_anomaly_area:
                continue
            
            # Extract bounding box
            min_row, min_col, max_row, max_col = region.bbox
            
            # Create region mask
            region_mask = (labeled_mask == region.label).astype(np.uint8)
            
            anomaly_region = {
                'bbox': (min_col, min_row, max_col - min_col, max_row - min_row),  # (x, y, w, h)
                'area': region.area,
                'centroid': (region.centroid[1], region.centroid[0]),  # (x, y)
                'mask': region_mask,
                'confidence': region.area / (binary_mask.shape[0] * binary_mask.shape[1])
            }
            
            anomaly_regions.append(anomaly_region)
        
        # Sort by area (largest first)
        anomaly_regions.sort(key=lambda x: x['area'], reverse=True)
        
        return anomaly_regions
    
    def localize_anomalies(
        self, 
        original: np.ndarray, 
        reconstruction: np.ndarray
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Localize anomalies in image.
        
        Args:
            original: Original image
            reconstruction: Reconstructed image
            
        Returns:
            Tuple of (error_map, anomaly_regions)
        """
        # Compute reconstruction error
        error_map = self.compute_reconstruction_error(original, reconstruction)
        
        # Threshold to create binary mask
        binary_mask = self.threshold_error_map(error_map)
        
        # Extract anomaly regions
        anomaly_regions = self.extract_anomaly_regions(binary_mask)
        
        return error_map, anomaly_regions


class AnomalyDetector(LoggerMixin):
    """
    Main anomaly detector class that combines model inference with localization.
    Provides comprehensive anomaly detection with scoring and visualization.
    """
    
    def __init__(
        self,
        model: ConvolutionalAutoencoder,
        localizer: Optional[AnomalyLocalizer] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize anomaly detector.
        
        Args:
            model: Trained autoencoder model
            localizer: Anomaly localizer instance
            device: Device for inference
        """
        self.model = model
        self.localizer = localizer or AnomalyLocalizer()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Statistics for threshold calculation
        self.reconstruction_errors = []
        self.threshold = None
        
        self.logger.info(f"Initialized anomaly detector on device: {self.device}")
    
    @handle_exceptions(PredictionError)
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            Preprocessed tensor [1, C, H, W]
        """
        # Normalize to [0, 1]
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        # Normalize using ImageNet statistics
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        
        return tensor.to(self.device)
    
    @handle_exceptions(PredictionError)
    def postprocess_reconstruction(self, reconstruction: torch.Tensor) -> np.ndarray:
        """
        Postprocess model reconstruction.
        
        Args:
            reconstruction: Model output tensor [1, C, H, W]
            
        Returns:
            Reconstructed image [H, W, C]
        """
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        reconstruction = reconstruction * std + mean
        
        # Convert to numpy and rearrange dimensions
        reconstruction = reconstruction.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # Clip to valid range
        reconstruction = np.clip(reconstruction, 0, 1)
        
        return reconstruction
    
    @handle_exceptions(PredictionError)
    def compute_anomaly_score(self, original: np.ndarray, reconstruction: np.ndarray) -> float:
        """
        Compute overall anomaly score for image.
        
        Args:
            original: Original image
            reconstruction: Reconstructed image
            
        Returns:
            Anomaly score (higher = more anomalous)
        """
        # Compute reconstruction error
        error = np.mean((original - reconstruction) ** 2)
        
        # Store for threshold calculation
        self.reconstruction_errors.append(error)
        
        return float(error)
    
    @handle_exceptions(PredictionError)
    def predict_single(self, image: np.ndarray) -> Dict:
        """
        Predict anomaly for single image.
        
        Args:
            image: Input image [H, W, C]
            
        Returns:
            Prediction dictionary
        """
        original_shape = image.shape[:2]
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Model inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            reconstruction_tensor = outputs['reconstruction']
        
        # Postprocess
        reconstruction = self.postprocess_reconstruction(reconstruction_tensor)
        
        # Resize reconstruction to match original
        if reconstruction.shape[:2] != original_shape:
            reconstruction = cv2.resize(
                reconstruction, 
                (original_shape[1], original_shape[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Compute anomaly score
        anomaly_score = self.compute_anomaly_score(image, reconstruction)
        
        # Localize anomalies
        error_map, anomaly_regions = self.localizer.localize_anomalies(image, reconstruction)
        
        # Determine if anomalous
        is_anomalous = self.is_anomalous(anomaly_score)
        
        return {
            'anomaly_score': anomaly_score,
            'is_anomalous': is_anomalous,
            'reconstruction': reconstruction,
            'error_map': error_map,
            'anomaly_regions': anomaly_regions,
            'num_anomalies': len(anomaly_regions)
        }
    
    def is_anomalous(self, anomaly_score: float) -> bool:
        """
        Determine if score indicates anomaly.
        
        Args:
            anomaly_score: Computed anomaly score
            
        Returns:
            True if anomalous
        """
        if self.threshold is None:
            # Use adaptive threshold based on collected scores
            if len(self.reconstruction_errors) > 10:
                self.threshold = np.percentile(self.reconstruction_errors, 95)
            else:
                return False  # Not enough data for threshold
        
        return anomaly_score > self.threshold
    
    def set_threshold(self, threshold: float):
        """Set anomaly threshold manually."""
        self.threshold = threshold
        self.logger.info(f"Anomaly threshold set to: {threshold}")
    
    def calibrate_threshold(self, good_images: List[np.ndarray], percentile: float = 95):
        """
        Calibrate threshold using good images.
        
        Args:
            good_images: List of known good images
            percentile: Percentile for threshold
        """
        self.logger.info(f"Calibrating threshold with {len(good_images)} good images")
        
        scores = []
        for image in good_images:
            result = self.predict_single(image)
            scores.append(result['anomaly_score'])
        
        self.threshold = np.percentile(scores, percentile)
        self.logger.info(f"Threshold calibrated to: {self.threshold}")
    
    def predict_batch(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Predict anomalies for batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for image in images:
            result = self.predict_single(image)
            results.append(result)
        
        return results


class AnomalyEvaluator(LoggerMixin):
    """
    Evaluates anomaly detection performance with comprehensive metrics.
    """
    
    def __init__(self):
        self.logger.info("Initialized anomaly evaluator")
    
    def evaluate_predictions(
        self, 
        predictions: List[Dict], 
        true_labels: List[int]
    ) -> Dict[str, float]:
        """
        Evaluate predictions against ground truth.
        
        Args:
            predictions: List of prediction dictionaries
            true_labels: List of true labels (0=good, 1=bad)
            
        Returns:
            Evaluation metrics dictionary
        """
        # Extract predicted labels and scores
        pred_labels = [int(pred['is_anomalous']) for pred in predictions]
        anomaly_scores = [pred['anomaly_score'] for pred in predictions]
        
        # Calculate metrics
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(true_labels, anomaly_scores)
        except ValueError:
            auc_roc = 0.0
        
        # Additional metrics
        tp = sum(1 for true, pred in zip(true_labels, pred_labels) if true == 1 and pred == 1)
        tn = sum(1 for true, pred in zip(true_labels, pred_labels) if true == 0 and pred == 0)
        fp = sum(1 for true, pred in zip(true_labels, pred_labels) if true == 0 and pred == 1)
        fn = sum(1 for true, pred in zip(true_labels, pred_labels) if true == 1 and pred == 0)
        
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        # Log metrics
        self.logger.info("Evaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")
        
        return metrics
    
    def print_classification_report(self, predictions: List[Dict], true_labels: List[int]):
        """Print detailed classification report."""
        from sklearn.metrics import classification_report
        
        pred_labels = [int(pred['is_anomalous']) for pred in predictions]
        
        report = classification_report(
            true_labels, 
            pred_labels, 
            target_names=['Good', 'Bad'],
            zero_division=0
        )
        
        print("\nClassification Report:")
        print(report)
