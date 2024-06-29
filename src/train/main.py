# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports --------------------------------------
# Add the parent directory to the sys.path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import pytorch lightning to train the model
import pytorch_lightning

# Import pandas to manipulate dataframes
import pandas

# Import model checkpoint (to store the trained model),
# the early stopping (to helo preventing overfitting) and the logger
# (to inform)
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

# Import wandb for the logger
import wandb

# Import sklearn to split randomly the dataset
from sklearn.model_selection import train_test_split

# Import transformers to get the tokenizer
from transformers import AutoTokenizer

# Import path to check files
from pathlib import Path

# Import argparse to use arguments
import argparse

# Import the created code
import config.configuration as config
from dataset_wrapper import CustomDatasetWrapper
from model import Model

# ----------------------------- Analyze arguments -----------------------------
def manage_arguments():
    # Get the arguments
    parser = argparse.ArgumentParser(
        description="Program to train and run the model")
    parser.add_argument('mode', nargs='?', type=int, default=1,
                        help="Tuning mode activation: \'0\' to deactivate," +
                        " \'1\' (default) to activate")
    args = parser.parse_args()
    
    # Set the mode in config
    if args.mode == 1: config.initialize_mode(1)
    else: config.initialize_mode(0)

# ----------------------------- Prepare the data ------------------------------
def prepare_data():
    # Split the dataset randomly into training and testing
    train_df, test_df = train_test_split(
        config.DATA, test_size=config.TEST_SIZE)
    # Store the dataframes in a csv file each
    train_df.to_csv(config.TRAIN_DATASET)
    test_df.to_csv(config.TEST_DATASET)

    # Create a tokenizer based on the model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
   
    # Create the wrapper
    dataset_wrapper = CustomDatasetWrapper(
      train_df,
      test_df,
      tokenizer,
      batch_size=config.BATCH_SIZE,
      max_token_len=config.MAX_TOKEN_COUNT
    )

    # Return the wrapper
    return dataset_wrapper

# ----------------------------- Prepare the model -----------------------------
def prepare_model(entry_count : int):
    # Calculate the steps
    steps_per_epoch = entry_count // config.BATCH_SIZE
    total_training_steps = steps_per_epoch * config.N_EPOCHS
    warmup_steps = total_training_steps // 5

    # Create the model
    model = Model(
      n_classes=len(config.CLASS_NAMES),
      n_warmup_steps=warmup_steps,
      n_training_steps=total_training_steps
    )

    # return the model
    return model

# ------------------------- Set wandb -----------------------------------------
def set_wandb():
    wandb.init(
      # set the wandb project where this run will be logged
      project=config.PROJECT_NAME,

      # track hyperparameters and run metadata
      config={
      "learning_rate": config.LEARNING_RATE,
      "architecture": "Bert",
      "dataset": config.DATASET,
      "epochs": config.N_EPOCHS,
    }
    )

    # Create an id for the log
    id = (config.MODEL_NAME + "_" + wandb.util.generate_id())

    # Create the logger
    logger = WandbLogger(project=config.PROJECT_NAME, id=id, resume="allow")

    return logger

# ------------------------- Train and test the model --------------------------
def main() -> None:
    # Prepare the data
    dataset_wrapper = prepare_data()
    entry_count = len(dataset_wrapper.train_df)

    # Prepare the model
    model = prepare_model(entry_count)

    # Save the trained model in checkpoints
    checkpoint_callback = ModelCheckpoint(
      dirpath="checkpoints",
      filename="best-checkpoint",
      save_top_k=1,
      verbose=True,
      monitor="val_loss",
      mode="min"
    )

    # Establish an early stopping once the validation loss has not changed
    # for the patience number of epochs
    early_stopping_callback = EarlyStopping(monitor='val_loss',
                                            patience=config.PATIENCE)

    # Create a log to store important data
    wandb_logger = set_wandb()
    
    base_logger = TensorBoardLogger("Logs", name="Noticias")
    
    # Create the trainer for the model
    trainer = pytorch_lightning.Trainer(
      logger=wandb_logger,
      callbacks=[checkpoint_callback, early_stopping_callback],
      max_epochs=config.N_EPOCHS,
      enable_progress_bar=True,
      log_every_n_steps=1
    )
    # Train the model
    trainer.fit(model, dataset_wrapper)
    # Test the model
    trainer.test(model, dataset_wrapper)

if __name__ == "__main__":
    # Manage the arguments
    manage_arguments()
    # Call main
    main()
