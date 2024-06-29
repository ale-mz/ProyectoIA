# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports ---------------------------------------
# Add the parent directory to the sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import argparse to use arguments
import argparse

# Import torch get running device (CPU/GPU)
import torch

# Import transformers to tokenize the input
from transformers import AutoTokenizer

# Import the created code
import config.configuration as config
from train.model import Model


# ----------------------------- Analyze arguments -----------------------------
def manage_arguments():
    # Get the arguments
    parser = argparse.ArgumentParser(
        description="Program to evaluate a trained model")
    parser.add_argument('input_path', nargs='?', type=str,
                        default=None,
                        help="Path to the txt file with the" +
                        " news to feed the model")
    args = parser.parse_args()
    
    return args.input_path

# ----------------------------- Load Model ------------------------------------
def load_model(device):
    # Load the trained model and move it to the appropriate device
    trained_model = Model.load_from_checkpoint(
        config.CHECKPOINT,
        n_classes=config.CLASS_NUM,
        n_warmup_steps=0,
        n_training_steps=0
    ).to(device)

    trained_model.eval()
    trained_model.freeze()

    return trained_model

# ----------------------------- Tokenize New ----------------------------------
def tokenize_new(new):
    # Create a tokenizer based on the model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Tokenize the input text
    encoding = tokenizer.encode_plus(
        new,
        add_special_tokens=True,
        max_length=config.MAX_TOKEN_COUNT,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )

    return encoding

# ----------------------------- Evaluate model --------------------------------
def main(file_path: str) -> None:
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            # Save the content of the file to news
            news = file.read().strip()
    except FileNotFoundError:
        print(f"Error: The input file {file_path} was not found.")
        return

    # Load the model from its checkpoint
    trained_model = load_model(device)
    # Tokenize the news from the file
    encoding = tokenize_new(news)

    # Move the input tensors to the appropriate device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Perform the prediction using the trained model
    _, test_prediction = trained_model(input_ids, attention_mask)
    test_prediction = test_prediction.flatten().cpu().numpy()

    # Print the labels with predictions above the threshold
    for label, prediction in zip(config.CLASS_NAMES, test_prediction):
        if prediction < config.THRESHOLD:
            continue
        print(f"{label}: {prediction}")

if __name__ == "__main__":
    # Manage the arguments
    file_path = manage_arguments()

    # If there were arguments
    if file_path is not None:
      # Call main
      main(file_path)