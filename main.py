"""
Main entry point for the Anomaly Detection System CLI.

This script provides a command-line interface to train, evaluate, and use the
anomaly detection model for identifying scratches on text images.

To use the CLI, run this script with Python and provide the desired command
and options. For example:

  # Train the model
  python main.py train --epochs 50 --batch-size 32

  # Evaluate the model
  python main.py evaluate --model-path models/best_model.pth

  # Predict on a single image
  python main.py predict --model-path models/best_model.pth --input /path/to/image.png

For more information on available commands and options, run:
  python main.py --help
"""

from src.interface import cli

if __name__ == '__main__':
    # The main entry point for the command-line interface.
    # The `cli` object is a Click command group that dispatches to the
    # appropriate command function (train, evaluate, predict, etc.) based on
    # the command-line arguments.
    cli()
