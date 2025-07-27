"""
Visualization module for anomaly detection results.
Provides comprehensive visualization of anomalies, error maps, and bounding boxes.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import seaborn as sns

from ..utils import LoggerMixin, VisualizationError, handle_exceptions


class AnomalyVisualizer(LoggerMixin):
    """
    Comprehensive visualizer for anomaly detection results.
    Supports various visualization modes and customization options.
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (15, 10),
        dpi: int = 100,
        colormap: str = 'hot',
        bbox_color: str = 'red',
        bbox_thickness: int = 2
    ):
        """
        Initialize visualizer.
        
        Args:
            figsize: Figure size for plots
            dpi: DPI for saved figures
            colormap: Colormap for error maps
            bbox_color: Color for bounding boxes
            bbox_thickness: Thickness of bounding box lines
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colormap = colormap
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info("Initialized anomaly visualizer")
    
    @handle_exceptions(VisualizationError)
    def plot_single_result(
        self,
        original: np.ndarray,
        result: Dict,
        title: Optional[str] = None,
        save_path: Optional[Path] = None,
        show_regions: bool = True,
        show_error_map: bool = True
    ) -> plt.Figure:
        """
        Plot comprehensive results for single image.
        
        Args:
            original: Original image
            result: Prediction result dictionary
            title: Optional title for the plot
            save_path: Optional path to save figure
            show_regions: Whether to show anomaly regions
            show_error_map: Whether to show error map
            
        Returns:
            Matplotlib figure
        """
        reconstruction = result['reconstruction']
        error_map = result['error_map']
        anomaly_regions = result['anomaly_regions']
        anomaly_score = result['anomaly_score']
        is_anomalous = result['is_anomalous']
        
        # Determine subplot layout
        n_cols = 3 if show_error_map else 2
        if show_regions and anomaly_regions:
            n_cols += 1
        
        fig, axes = plt.subplots(1, n_cols, figsize=self.figsize, dpi=self.dpi)
        if n_cols == 1:
            axes = [axes]
        
        col_idx = 0
        
        # Original image
        axes[col_idx].imshow(original)
        axes[col_idx].set_title('Original Image')
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Reconstruction
        axes[col_idx].imshow(reconstruction)
        axes[col_idx].set_title(f'Reconstruction\nScore: {anomaly_score:.4f}')
        axes[col_idx].axis('off')
        col_idx += 1
        
        # Error map
        if show_error_map:
            im = axes[col_idx].imshow(error_map, cmap=self.colormap)
            axes[col_idx].set_title('Error Map')
            axes[col_idx].axis('off')
            plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)
            col_idx += 1
        
        # Anomaly regions
        if show_regions and anomaly_regions:
            # Show original with bounding boxes
            axes[col_idx].imshow(original)
            
            for region in anomaly_regions:
                bbox = region['bbox']  # (x, y, w, h)
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    linewidth=self.bbox_thickness,
                    edgecolor=self.bbox_color,
                    facecolor='none'
                )
                axes[col_idx].add_patch(rect)
                
                # Add confidence text
                axes[col_idx].text(
                    bbox[0], bbox[1] - 5,
                    f'{region["confidence"]:.3f}',
                    color=self.bbox_color,
                    fontsize=8,
                    weight='bold'
                )
            
            axes[col_idx].set_title(f'Detected Regions ({len(anomaly_regions)})')
            axes[col_idx].axis('off')
        
        # Overall title
        status = "ANOMALOUS" if is_anomalous else "NORMAL"
        main_title = title or f"Anomaly Detection Result: {status}"
        fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    @handle_exceptions(VisualizationError)
    def plot_batch_results(
        self,
        images: List[np.ndarray],
        results: List[Dict],
        titles: Optional[List[str]] = None,
        max_images: int = 12,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot results for batch of images in grid layout.
        
        Args:
            images: List of original images
            results: List of prediction results
            titles: Optional titles for each image
            max_images: Maximum number of images to show
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_images = min(len(images), max_images)
        n_cols = min(4, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(n_cols * 4, n_rows * 4), 
            dpi=self.dpi
        )
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        if n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(n_images):
            row = i // n_cols
            col = i % n_cols
            
            image = images[i]
            result = results[i]
            
            # Show original with bounding boxes if anomalous
            axes[row, col].imshow(image)
            
            if result['is_anomalous'] and result['anomaly_regions']:
                for region in result['anomaly_regions']:
                    bbox = region['bbox']
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), bbox[2], bbox[3],
                        linewidth=2,
                        edgecolor=self.bbox_color,
                        facecolor='none'
                    )
                    axes[row, col].add_patch(rect)
            
            # Title with status and score
            status = "ANOMALOUS" if result['is_anomalous'] else "NORMAL"
            title = titles[i] if titles else f"Image {i+1}"
            axes[row, col].set_title(
                f"{title}\n{status} ({result['anomaly_score']:.3f})",
                fontsize=10
            )
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(n_images, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Batch results saved to {save_path}")
        
        return fig
    
    @handle_exceptions(VisualizationError)
    def plot_error_distribution(
        self,
        results: List[Dict],
        true_labels: Optional[List[int]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot distribution of anomaly scores.
        
        Args:
            results: List of prediction results
            true_labels: Optional true labels for coloring
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        scores = [result['anomaly_score'] for result in results]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
        
        # Histogram
        if true_labels is not None:
            good_scores = [score for score, label in zip(scores, true_labels) if label == 0]
            bad_scores = [score for score, label in zip(scores, true_labels) if label == 1]
            
            axes[0].hist(good_scores, bins=30, alpha=0.7, label='Good Images', color='green')
            axes[0].hist(bad_scores, bins=30, alpha=0.7, label='Bad Images', color='red')
            axes[0].legend()
        else:
            axes[0].hist(scores, bins=30, alpha=0.7, color='blue')
        
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Anomaly Scores')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        if true_labels is not None:
            data = [good_scores, bad_scores]
            labels = ['Good', 'Bad']
            colors = ['green', 'red']
        else:
            data = [scores]
            labels = ['All Images']
            colors = ['blue']
        
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1].set_ylabel('Anomaly Score')
        axes[1].set_title('Anomaly Score Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Error distribution plot saved to {save_path}")
        
        return fig
    
    @handle_exceptions(VisualizationError)
    def plot_roc_curve(
        self,
        results: List[Dict],
        true_labels: List[int],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot ROC curve for anomaly detection.
        
        Args:
            results: List of prediction results
            true_labels: True labels
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import roc_curve, auc
        
        scores = [result['anomaly_score'] for result in results]
        
        fpr, tpr, thresholds = roc_curve(true_labels, scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"ROC curve saved to {save_path}")
        
        return fig
    
    @handle_exceptions(VisualizationError)
    def plot_precision_recall_curve(
        self,
        results: List[Dict],
        true_labels: List[int],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            results: List of prediction results
            true_labels: True labels
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        scores = [result['anomaly_score'] for result in results]
        
        precision, recall, thresholds = precision_recall_curve(true_labels, scores)
        avg_precision = average_precision_score(true_labels, scores)
        
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"PR curve saved to {save_path}")
        
        return fig
    
    @handle_exceptions(VisualizationError)
    def create_anomaly_heatmap(
        self,
        original: np.ndarray,
        error_map: np.ndarray,
        alpha: float = 0.6,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create heatmap overlay of anomaly regions.
        
        Args:
            original: Original image
            error_map: Error map
            alpha: Transparency of overlay
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        # Show original image
        ax.imshow(original)
        
        # Overlay error map
        im = ax.imshow(error_map, cmap=self.colormap, alpha=alpha)
        
        ax.set_title('Anomaly Heatmap Overlay')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Error Intensity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Anomaly heatmap saved to {save_path}")
        
        return fig
    
    def show_all_plots(self):
        """Display all created plots."""
        plt.show()
    
    def close_all_plots(self):
        """Close all plots to free memory."""
        plt.close('all')


class InteractiveVisualizer(AnomalyVisualizer):
    """
    Interactive visualizer with threshold adjustment capabilities.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_threshold = None
        self.current_results = None
        self.current_images = None
    
    def create_threshold_slider(
        self,
        images: List[np.ndarray],
        detector,
        initial_threshold: float = 0.01,
        threshold_range: Tuple[float, float] = (0.001, 0.1)
    ):
        """
        Create interactive threshold adjustment interface.
        
        Args:
            images: List of images to test
            detector: Anomaly detector instance
            initial_threshold: Initial threshold value
            threshold_range: Range of threshold values
        """
        try:
            from ipywidgets import interact, FloatSlider
            import IPython.display as display
        except ImportError:
            self.logger.warning("Interactive widgets not available. Install ipywidgets for interactive features.")
            return
        
        self.current_images = images
        detector.set_threshold(initial_threshold)
        
        def update_visualization(threshold):
            detector.set_threshold(threshold)
            results = detector.predict_batch(images)
            
            # Count anomalies
            n_anomalies = sum(1 for r in results if r['is_anomalous'])
            
            print(f"Threshold: {threshold:.4f} | Detected anomalies: {n_anomalies}/{len(images)}")
            
            # Show sample results
            if len(images) <= 4:
                fig = self.plot_batch_results(images, results)
                plt.show()
        
        # Create slider
        threshold_slider = FloatSlider(
            value=initial_threshold,
            min=threshold_range[0],
            max=threshold_range[1],
            step=(threshold_range[1] - threshold_range[0]) / 100,
            description='Threshold:',
            continuous_update=False
        )
        
        interact(update_visualization, threshold=threshold_slider)
