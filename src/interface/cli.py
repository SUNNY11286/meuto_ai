"""
Command Line Interface for the anomaly detection system.
Provides comprehensive CLI for training, evaluation, and prediction.
"""

import click
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List
from omegaconf import OmegaConf
import json
from datetime import datetime

from ..utils import ConfigManager, LoggerManager, setup_logging
from ..data import DatasetBuilder
from ..models import create_autoencoder
from ..training import AnomalyDetectionTrainer
from ..detection import AnomalyDetector, AnomalyLocalizer, AnomalyEvaluator, AnomalyVisualizer


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), 
              default='config/config.yaml', help='Configuration file path')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """Anomaly Detection System CLI"""
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(level=log_level)
    
    # Load configuration
    config_manager = ConfigManager(config)
    ctx.obj['config'] = config_manager.config
    ctx.obj['config_manager'] = config_manager
    
    click.echo(f"Loaded configuration from: {config}")


@cli.command()
@click.option('--epochs', '-e', type=int, help='Number of training epochs')
@click.option('--batch-size', '-b', type=int, help='Batch size for training')
@click.option('--learning-rate', '-lr', type=float, help='Learning rate')
@click.option('--device', type=click.Choice(['auto', 'cpu', 'cuda']), 
              default='auto', help='Device to use for training')
@click.option('--resume', type=click.Path(exists=True), help='Resume from checkpoint')
@click.pass_context
def train(ctx, epochs, batch_size, learning_rate, device, resume):
    """Train the anomaly detection model"""
    config = ctx.obj['config']
    
    # Override config with CLI arguments
    if epochs:
        config.training.epochs = epochs
    if batch_size:
        config.data.batch_size = batch_size
    if learning_rate:
        config.training.learning_rate = learning_rate
    
    # Setup device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    click.echo(f"Training on device: {device}")
    
    try:
        # Create datasets
        click.echo("Preparing datasets...")
        data_config = OmegaConf.to_container(config.data, resolve=True)
        data_config['data_dir'] = data_config.pop('dataset_path')
        # Pop keys not used by DatasetBuilder constructor
        data_config.pop('batch_size', None)
        data_config.pop('num_workers', None)
        dataset_builder = DatasetBuilder(**data_config)
        datasets = dataset_builder.build_datasets()
        dataloaders = dataset_builder.build_dataloaders(datasets, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
        
        # Create model
        click.echo("Creating model...")
        model = create_autoencoder(config)
        
        # Create trainer
        trainer = AnomalyDetectionTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            device=device,
            config=config.training.__dict__
        )
        
        # Resume from checkpoint if specified
        if resume:
            click.echo(f"Resuming from checkpoint: {resume}")
            trainer.load_model(Path(resume))
        
        # Train model
        click.echo("Starting training...")
        history = trainer.train()
        
        # Save final model
        model_path = Path(config.paths.models_dir) / 'final_model.pth'
        trainer.save_model(model_path)
        click.echo(f"Model saved to: {model_path}")
        
        # Plot training history
        plot_path = Path(config.paths.results_dir) / 'training_history.png'
        trainer.plot_training_history(plot_path)
        
        click.echo("Training completed successfully!")
        
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--model-path', '-m', type=click.Path(exists=True), 
              required=True, help='Path to trained model')
@click.option('--test-split', type=click.Choice(['test', 'val']), 
              default='test', help='Which split to evaluate on')
@click.option('--threshold', '-t', type=float, help='Anomaly threshold')
@click.option('--save-results', is_flag=True, help='Save evaluation results')
@click.pass_context
def evaluate(ctx, model_path, test_split, threshold, save_results):
    """Evaluate the trained model"""
    config = ctx.obj['config']
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        click.echo(f"Loading model from: {model_path}")
        model = create_autoencoder(config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create datasets
        click.echo("Preparing datasets...")
        dataset_builder = DatasetBuilder(config)
        datasets = dataset_builder.build_datasets()
        
        # Get test dataset
        test_dataset = datasets[test_split]
        
        # Create detector
        localizer = AnomalyLocalizer()
        detector = AnomalyDetector(model, localizer, device)
        
        # Set threshold if provided
        if threshold:
            detector.set_threshold(threshold)
        else:
            # Calibrate threshold using good images
            click.echo("Calibrating threshold...")
            good_images = []
            for i, sample in enumerate(test_dataset):
                if sample['label'] == 0 and len(good_images) < 100:  # Good images
                    image = sample['image'].permute(1, 2, 0).numpy()
                    good_images.append(image)
            detector.calibrate_threshold(good_images)
        
        # Run evaluation
        click.echo("Running evaluation...")
        images = []
        true_labels = []
        
        for sample in test_dataset:
            image = sample['image'].permute(1, 2, 0).numpy()
            images.append(image)
            true_labels.append(sample['label'])
        
        # Predict
        results = detector.predict_batch(images)
        
        # Evaluate
        evaluator = AnomalyEvaluator()
        metrics = evaluator.evaluate_predictions(results, true_labels)
        evaluator.print_classification_report(results, true_labels)
        
        # Visualize results
        visualizer = AnomalyVisualizer()
        
        # Plot error distribution
        dist_plot = visualizer.plot_error_distribution(results, true_labels)
        if save_results:
            dist_path = Path(config.paths.results_dir) / 'error_distribution.png'
            dist_plot.savefig(dist_path)
        
        # Plot ROC curve
        roc_plot = visualizer.plot_roc_curve(results, true_labels)
        if save_results:
            roc_path = Path(config.paths.results_dir) / 'roc_curve.png'
            roc_plot.savefig(roc_path)
        
        # Show sample results
        sample_indices = np.random.choice(len(images), min(8, len(images)), replace=False)
        sample_images = [images[i] for i in sample_indices]
        sample_results = [results[i] for i in sample_indices]
        
        batch_plot = visualizer.plot_batch_results(sample_images, sample_results)
        if save_results:
            batch_path = Path(config.paths.results_dir) / 'sample_results.png'
            batch_plot.savefig(batch_path)
        
        visualizer.show_all_plots()
        
        # Save metrics if requested
        if save_results:
            metrics_path = Path(config.paths.results_dir) / 'evaluation_metrics.json'
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            click.echo(f"Results saved to: {config.paths.results_dir}")
        
        click.echo("Evaluation completed successfully!")
        
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--model-path', '-m', type=click.Path(exists=True), 
              required=True, help='Path to trained model')
@click.option('--input', '-i', type=click.Path(exists=True), 
              required=True, help='Input image or directory')
@click.option('--output', '-o', type=click.Path(), help='Output directory for results')
@click.option('--threshold', '-t', type=float, help='Anomaly threshold')
@click.option('--visualize', is_flag=True, help='Show visualization')
@click.option('--save-images', is_flag=True, help='Save result images')
@click.pass_context
def predict(ctx, model_path, input, output, threshold, visualize, save_images):
    """Predict anomalies in images"""
    config = ctx.obj['config']
    
    try:
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        click.echo(f"Loading model from: {model_path}")
        model = create_autoencoder(config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create detector
        localizer = AnomalyLocalizer()
        detector = AnomalyDetector(model, localizer, device)
        
        # Set threshold if provided
        if threshold:
            detector.set_threshold(threshold)
        
        # Prepare input
        input_path = Path(input)
        if input_path.is_file():
            image_paths = [input_path]
        else:
            # Find all images in directory
            extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            image_paths = []
            for ext in extensions:
                image_paths.extend(input_path.glob(f'*{ext}'))
                image_paths.extend(input_path.glob(f'*{ext.upper()}'))
        
        if not image_paths:
            click.echo("No images found in input path", err=True)
            raise click.Abort()
        
        click.echo(f"Processing {len(image_paths)} images...")
        
        # Process images
        results = []
        for img_path in image_paths:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                click.echo(f"Failed to load image: {img_path}", err=True)
                continue
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            # Predict
            result = detector.predict_single(image)
            result['image_path'] = str(img_path)
            result['image'] = image
            results.append(result)
            
            # Print result
            status = "ANOMALOUS" if result['is_anomalous'] else "NORMAL"
            click.echo(f"{img_path.name}: {status} (score: {result['anomaly_score']:.4f}, "
                      f"regions: {result['num_anomalies']})")
        
        # Create output directory if needed
        if output or save_images:
            output_dir = Path(output) if output else Path('predictions')
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualize results
        if visualize or save_images:
            visualizer = AnomalyVisualizer()
            
            for i, result in enumerate(results):
                img_path = Path(result['image_path'])
                
                # Create visualization
                fig = visualizer.plot_single_result(
                    result['image'], 
                    result,
                    title=f"{img_path.name}"
                )
                
                if save_images:
                    save_path = output_dir / f"{img_path.stem}_result.png"
                    fig.savefig(save_path, dpi=150, bbox_inches='tight')
                    click.echo(f"Saved result to: {save_path}")
                
                if not save_images and visualize:
                    visualizer.show_all_plots()
        
        # Save summary
        if output:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(model_path),
                'threshold': detector.threshold,
                'total_images': len(results),
                'anomalous_images': sum(1 for r in results if r['is_anomalous']),
                'results': [
                    {
                        'image_path': r['image_path'],
                        'is_anomalous': r['is_anomalous'],
                        'anomaly_score': r['anomaly_score'],
                        'num_regions': r['num_anomalies']
                    }
                    for r in results
                ]
            }
            
            summary_path = output_dir / 'prediction_summary.json'
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            click.echo(f"Summary saved to: {summary_path}")
        
        click.echo("Prediction completed successfully!")
        
    except Exception as e:
        click.echo(f"Prediction failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information"""
    config = ctx.obj['config']
    
    click.echo("=== Anomaly Detection System Information ===")
    click.echo(f"PyTorch version: {torch.__version__}")
    click.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        click.echo(f"CUDA device: {torch.cuda.get_device_name()}")
    
    click.echo(f"\nData directory: {config.data.data_dir}")
    click.echo(f"Models directory: {config.paths.models_dir}")
    click.echo(f"Results directory: {config.paths.results_dir}")
    
    # Check data availability
    data_dir = Path(config.data.data_dir)
    if data_dir.exists():
        good_dir = data_dir / 'good'
        bad_dir = data_dir / 'bad'
        masks_dir = data_dir / 'masks'
        
        if good_dir.exists():
            good_count = len(list(good_dir.glob('*.png')))
            click.echo(f"Good images: {good_count}")
        
        if bad_dir.exists():
            bad_count = len(list(bad_dir.glob('*.png')))
            click.echo(f"Bad images: {bad_count}")
        
        if masks_dir.exists():
            mask_count = len(list(masks_dir.glob('*.png')))
            click.echo(f"Mask images: {mask_count}")
    else:
        click.echo("Data directory not found!")


if __name__ == '__main__':
    cli()
