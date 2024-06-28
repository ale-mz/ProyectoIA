# ----------------------------- Imports ---------------------------------------
# Add the parent directory to the sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import torch get running device (CPU/GPU)
import torch

# Import transformers to tokenize the input
from transformers import AutoTokenizer

# Import the created code
import configuration.configuration as config
from train.model import Model


# ----------------------------- Load Model ---------------------------------------
def load_model(device):

    # Load the checkpoint
    checkpoint_path = config.CHECKPOINT

    # Load the trained model and move it to the appropriate device
    trained_model = Model.load_from_checkpoint(
        checkpoint_path,
        n_classes=config.CLASS_NUM,
        n_warmup_steps=0,
        n_training_steps=0
    ).to(device)

    trained_model.eval()
    trained_model.freeze()

    return trained_model

# ----------------------------- Tokenize New ---------------------------------------
def tokenize_new(new):
    # Create a tokenizer based on the model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Encode the input text
    encoding = tokenizer.encode_plus(
        new,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        return_attention_mask=True,
        return_tensors='pt',
    )
    return encoding

def main(file_path: str) -> None:
    # Set device to GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Read the content of the file
        with open(file_path, 'r', encoding='utf-8') as file:
            news = file.read().strip()  # Save the content of the file to news
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return

    trained_model = load_model(device)

    encoding = tokenize_new(news)

    # Move the input tensors to the appropriate device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Perform prediction using a trained model
    _, test_prediction = trained_model(input_ids, attention_mask)
    test_prediction = test_prediction.flatten().cpu().numpy()

    # Print the labels with predictions above the threshold
    for label, prediction in zip(config.CLASS_NAMES, test_prediction):
        if prediction < config.THRESHOLD:
            continue
        print(f"{label}: {prediction}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python label_evaluator.py input_file.txt")
    else:
        file_path = sys.argv[1]
        main(file_path)