# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports ---------------------------------------
import pytorch_lightning
from torch.utils.data import DataLoader

from dataset import CustomDataset

# ----------------------------- Class definition ------------------------------
# Wrapper class for the custom dataset
class CustomDatasetWrapper(pytorch_lightning.LightningDataModule):
  # Constructor
  def __init__(self, train_df, test_df, tokenizer, batch_size=8,
               max_token_len=512):
    super().__init__()
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_token_len = max_token_len

  # Override method to create both datasets
  def setup(self, stage=None):
    self.train_dataset = CustomDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )
    self.test_dataset = CustomDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  # Method to load the data for training
  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2,
      pin_memory=True
    )

  # Method to load the data for validating
  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2, 
      pin_memory=True
    )

  # Method to load the data for testing
  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2, 
      pin_memory=True
    )