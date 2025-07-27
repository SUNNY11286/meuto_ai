"""
Dataset classes for loading and preprocessing text images with scratch detection.
Implements efficient data loading with augmentation and validation.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Callable
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

from ..utils import LoggerMixin, DataLoadingError, handle_exceptions


class TextImageDataset(Dataset, LoggerMixin):
    """
    Dataset class for loading text images with optional masks for anomaly detection.
    Supports both good and bad images with corresponding masks.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        image_paths: List[str],
        labels: List[int],
        mask_paths: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        is_training: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Root directory containing images
            image_paths: List of image file paths
            labels: List of labels (0 for good, 1 for bad)
            mask_paths: Optional list of mask file paths
            transform: Optional transform to apply
            image_size: Target image size (height, width)
            is_training: Whether this is training data
        """
        self.data_dir = Path(data_dir)
        self.image_paths = image_paths
        self.labels = labels
        self.mask_paths = mask_paths or [None] * len(image_paths)
        self.transform = transform
        self.image_size = image_size
        self.is_training = is_training
        
        self.logger.info(f"Initialized dataset with {len(self.image_paths)} images")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.image_paths)
    
    @handle_exceptions(DataLoadingError)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index of item to retrieve
            
        Returns:
            Dictionary containing image, label, and optionally mask
        """
        # Load image
        image_path = self.data_dir / self.image_paths[idx]
        image = self._load_image(image_path)
        
        # Load mask if available
        mask = None
        if self.mask_paths[idx] is not None:
            mask_path = self.data_dir / "masks" / self.mask_paths[idx]
            if mask_path.exists():
                mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            if mask is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            else:
                transformed = self.transform(image=image)
                image = transformed['image']
        
        # Prepare output
        output = {
            'image': image,
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'image_path': str(image_path)
        }
        
        if mask is not None:
            output['mask'] = mask
        
        return output
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load and preprocess image."""
        try:
            # Load image using OpenCV for better performance
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
            
            return image
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load image {image_path}: {e}")
    
    def _load_mask(self, mask_path: Path) -> np.ndarray:
        """Load and preprocess mask."""
        try:
            # Load mask as grayscale
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {mask_path}")
            
            # Resize mask
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)
            
            # Normalize to 0-1
            mask = mask.astype(np.float32) / 255.0
            
            return mask
            
        except Exception as e:
            raise DataLoadingError(f"Failed to load mask {mask_path}: {e}")


class DatasetBuilder(LoggerMixin):
    """
    Builder class for creating datasets with proper train/validation/test splits.
    Handles data discovery, splitting, and augmentation setup.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        good_folder: str = "good",
        bad_folder: str = "bad",
        masks_folder: str = "masks",
        image_size: Tuple[int, int] = (256, 256),
        test_split: float = 0.1,
        val_split: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize dataset builder.
        
        Args:
            data_dir: Root directory containing data
            good_folder: Folder name for good images
            bad_folder: Folder name for bad images
            masks_folder: Folder name for masks
            image_size: Target image size
            test_split: Fraction of data for testing
            val_split: Fraction of data for validation
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.good_folder = good_folder
        self.bad_folder = bad_folder
        self.masks_folder = masks_folder
        self.image_size = image_size
        self.test_split = test_split
        self.val_split = val_split
        self.random_state = random_state
        
        self.logger.info(f"Initialized dataset builder for {self.data_dir}")
    
    @handle_exceptions(DataLoadingError)
    def discover_data(self) -> Tuple[List[str], List[int], List[str]]:
        """
        Discover all images and their corresponding labels and masks.
        
        Returns:
            Tuple of (image_paths, labels, mask_paths)
        """
        image_paths = []
        labels = []
        mask_paths = []
        
        # Process good images
        good_dir = self.data_dir / self.good_folder
        if good_dir.exists():
            good_images = list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))
            for img_path in good_images:
                image_paths.append(f"{self.good_folder}/{img_path.name}")
                labels.append(0)  # 0 for good images
                mask_paths.append(None)  # No masks for good images
        
        # Process bad images
        bad_dir = self.data_dir / self.bad_folder
        masks_dir = self.data_dir / self.masks_folder
        
        if bad_dir.exists():
            bad_images = list(bad_dir.glob("*.png")) + list(bad_dir.glob("*.jpg"))
            for img_path in bad_images:
                image_paths.append(f"{self.bad_folder}/{img_path.name}")
                labels.append(1)  # 1 for bad images
                
                # Check for corresponding mask
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    mask_paths.append(img_path.name)
                else:
                    mask_paths.append(None)
        
        self.logger.info(f"Discovered {len(image_paths)} images: "
                        f"{labels.count(0)} good, {labels.count(1)} bad")
        
        return image_paths, labels, mask_paths
    
    @handle_exceptions(DataLoadingError)
    def create_splits(
        self,
        image_paths: List[str],
        labels: List[int],
        mask_paths: List[str]
    ) -> Dict[str, Tuple[List[str], List[int], List[str]]]:
        """
        Create train/validation/test splits.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            mask_paths: List of mask paths
            
        Returns:
            Dictionary with train/val/test splits
        """
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels, train_val_masks, test_masks = \
            train_test_split(
                image_paths, labels, mask_paths,
                test_size=self.test_split,
                stratify=labels,
                random_state=self.random_state
            )
        
        # Second split: separate validation from training
        if self.val_split > 0:
            train_paths, val_paths, train_labels, val_labels, train_masks, val_masks = \
                train_test_split(
                    train_val_paths, train_val_labels, train_val_masks,
                    test_size=self.val_split / (1 - self.test_split),
                    stratify=train_val_labels,
                    random_state=self.random_state
                )
        else:
            train_paths, train_labels, train_masks = train_val_paths, train_val_labels, train_val_masks
            val_paths, val_labels, val_masks = [], [], []
        
        splits = {
            'train': (train_paths, train_labels, train_masks),
            'val': (val_paths, val_labels, val_masks),
            'test': (test_paths, test_labels, test_masks)
        }
        
        for split_name, (paths, lbls, _) in splits.items():
            good_count = lbls.count(0)
            bad_count = lbls.count(1)
            self.logger.info(f"{split_name.capitalize()} set: {len(paths)} images "
                           f"({good_count} good, {bad_count} bad)")
        
        return splits
    
    def get_transforms(self, is_training: bool = True) -> A.Compose:
        """
        Get augmentation transforms.
        
        Args:
            is_training: Whether to apply training augmentations
            
        Returns:
            Albumentations compose object
        """
        if is_training:
            transform = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.Blur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        
        return transform
    
    def build_datasets(self) -> Dict[str, TextImageDataset]:
        """
        Build train/validation/test datasets.
        
        Returns:
            Dictionary containing datasets
        """
        # Discover data
        image_paths, labels, mask_paths = self.discover_data()
        
        # Create splits
        splits = self.create_splits(image_paths, labels, mask_paths)
        
        # Build datasets
        datasets = {}
        
        for split_name, (paths, lbls, masks) in splits.items():
            if len(paths) == 0:
                continue
                
            is_training = split_name == 'train'
            transform = self.get_transforms(is_training)
            
            dataset = TextImageDataset(
                data_dir=self.data_dir,
                image_paths=paths,
                labels=lbls,
                mask_paths=masks,
                transform=transform,
                image_size=self.image_size,
                is_training=is_training
            )
            
            datasets[split_name] = dataset
        
        return datasets
    
    def build_dataloaders(
        self,
        datasets: Dict[str, TextImageDataset],
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Dict[str, DataLoader]:
        """
        Build data loaders from datasets.
        
        Args:
            datasets: Dictionary of datasets
            batch_size: Batch size for training
            num_workers: Number of worker processes
            
        Returns:
            Dictionary containing data loaders
        """
        dataloaders = {}
        
        for split_name, dataset in datasets.items():
            shuffle = split_name == 'train'
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=shuffle  # Drop last batch only for training
            )
            
            dataloaders[split_name] = dataloader
        
        return dataloaders
