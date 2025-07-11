# Multi-Label News Classification with BERT and PyTorch Lightning

> ⚠️ **Warning**  
> This project requires a **powerful device with a GPU** and the **appropriate CUDA drivers** installed for optimal performance. Training BERT-based models without GPU support may result in extremely slow training or execution failure.

## Overview

This project implements a **multi-label text classification** system to categorize news articles into predefined categories using a **fine-tuned BERT model**. Developed with **PyTorch Lightning** for efficient training and experiment management, this solution provides a robust and scalable approach to news categorization.

## Features

- **Multi-Label Classification**: Accurately assigns multiple relevant categories to a single news article.
- **BERT-based Model**: Leverages the power of pre-trained BERT models (`bert-base-multilingual-cased` by default) for high-performance text understanding.
- **PyTorch Lightning Framework**: Utilizes PyTorch Lightning for streamlined training loops, automatic device placement (GPU/CPU), and reduced boilerplate code.
- **Configurable Training**: Easily adjustable hyperparameters such as learning rate, batch size, number of epochs, and model name via `configuration.py`.
- **Early Stopping & Checkpointing**: Prevents overfitting and saves the best performing model checkpoint during training.
- **Comprehensive Metrics**: Tracks key multi-label classification metrics including Macro and Micro Accuracy, Precision, Recall, and F1-Score.
- **Weights & Biases Integration**: Logs experiment metrics, hyperparameters, and model checkpoints to Weights & Biases.
- **Inference Script**: Includes `label_evaluator.py` to classify new articles using a trained model.

## Project Structure
.
├── checkpoints/
│   └── best-checkpoint.ckpt  # Trained model checkpoints
├── config/
│   └── configuration.py      # Global project configurations and hyperparameters
├── datasets/
│   ├── Dataset_noticias.csv  # Raw input dataset (example)
│   ├── Train_Noticias.csv    # Processed training data
│   └── Test_Noticias.csv     # Processed testing data
├── dataset.py                # PyTorch Dataset for loading and tokenizing news data
├── dataset_wrapper.py        # PyTorch Lightning DataModule for data loaders
├── label_evaluator.py        # Script for evaluating news articles with a trained model
├── main.py                   # Main script for training and testing the model
└── model.py                  # PyTorch Lightning module defining the BERT model and its training logic

Getting Started

Prerequisites

    Python 3.8+

    pip package manager

Installation

    Clone the repository:
    Bash

git clone <your-repository-url>
cd <your-project-directory>

Install dependencies:
Bash

    pip install -r requirements.txt

    (A requirements.txt file listing pytorch-lightning, pandas, scikit-learn, transformers, wandb, torchmetrics, and torch would be needed here).

    Prepare your dataset:
    Place your multi-label news dataset (e.g., Dataset_noticias.csv) in the datasets/ directory as specified in config/configuration.py. Ensure it has a column for the news text and separate columns for each label, typically represented as binary (0/1) values.

Training the Model

To train the model, run main.py. The script will automatically split your dataset, prepare the data loaders, initialize the model, and start the training process.
Bash

python main.py

By default, the tuning mode is activated, setting N_EPOCHS to 1 and fixing the random seed for reproducibility.

To run a full training without the "tuning" mode (e.g., for more epochs), you can pass 0 as an argument:
Bash

python main.py 0

This will run training for N_EPOCHS as defined in configuration.py (default 30).

Evaluating a News Article

To evaluate a new news article using the trained model (the best-checkpoint.ckpt in the checkpoints directory), use the label_evaluator.py script.

    Create an input text file: Create a .txt file containing the news article you want to classify (e.g., my_news.txt).

    Run the evaluator:
    Bash

    python label_evaluator.py my_news.txt

    The script will print the predicted labels and their confidence scores above the configured THRESHOLD.

Configuration

All key configurations, including model choice, dataset paths, hyperparameters, and evaluation thresholds, are managed in config/configuration.py. This allows for easy customization and experimentation.

Technologies Used

    Python

    PyTorch Lightning

    Hugging Face Transformers

    PyTorch

    Pandas

    Scikit-learn

    Weights & Biases

    TorchMetrics

Authors

    Oscar Fernández

    Jorge Madríz

    Kenneth Villalobos

Acknowledgements

    Code adapted from:

        Multi-Label Text Classification with BERT and PyTorch Lightning by Curiousily

        Fine-Tune BERT for Text Classification by Pritish Mishra on Kaggle

    Special thanks to A. Badilla
