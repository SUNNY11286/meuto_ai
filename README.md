# Text Scratch Detection using Autoencoders

This project implements a production-grade anomaly detection system to identify scratches on text images using a deep convolutional autoencoder. The system is built with PyTorch and includes a command-line interface (CLI) for training, evaluation, and prediction.

## Features

- **Autoencoder Model**: A deep convolutional autoencoder for unsupervised anomaly detection.
- **Command-Line Interface**: A full-featured CLI for managing the entire workflow.
- **Configuration Management**: Centralized configuration using YAML files (`config/config.yaml`).
- **Structured Logging**: Detailed logging with `loguru`, including file and console outputs.
- **Data Handling**: Efficient data loading and preprocessing with `torch.utils.data` and `albumentations`.
- **Training and Evaluation**: Includes training, validation, and testing loops with metrics.
- **Production-Grade Code**: Follows modern software engineering practices, including OOP, exception handling, and a modular structure.

## Project Structure

```
text-scratch-detection/
├── config/
│   └── config.yaml         # Configuration file
├── anomaly_detection_test_data/ # Dataset directory (ignored by Git)
│   ├── good/
│   ├── bad/
│   └── masks/
├── src/
│   ├── data/               # Data loading and processing
│   ├── detection/          # Anomaly detection and visualization
│   ├── interface/          # Command-line interface
│   ├── models/             # Model architectures
│   ├── training/           # Training and evaluation logic
│   └── utils/              # Utilities (config, logger, etc.)
├── main.py                 # Main entry point for the CLI
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

1.  **Clone the repository (once pushed to GitHub):**
    ```bash
    git clone <your-repository-url>
    cd text-scratch-detection
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is controlled via the `main.py` script. You can see all available commands by running:

```bash
python main.py --help
```

### Training

To train the autoencoder model, run the `train` command:

```bash
python main.py train
```

You can override configuration parameters from the command line:

```bash
python main.py train --epochs 50 --learning-rate 0.0005 --batch-size 16
```

### Evaluation

To evaluate the trained model on the test set:

```bash
python main.py evaluate --model-path models/best_model.pth
```

### Prediction

To run anomaly detection on a single image or a directory of images:

```bash
# Single image
python main.py predict --model-path models/best_model.pth --input-path /path/to/your/image.png

# Directory of images
python main.py predict --model-path models/best_model.pth --input-path /path/to/your/directory/
```

### Display Info

To display information about the dataset:

```bash
python main.py info
```
