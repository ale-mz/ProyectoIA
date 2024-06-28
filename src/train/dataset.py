# Copyright Oscar Fernández, Jorge Madríz y Kenneth Villalobos
# Code adapted from: https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/
# Code adapted from: https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029
# Special thanks to A. Badilla

# ----------------------------- Imports ---------------------------------------
import pandas

import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import configuration.configuration as config

# ----------------------------- Class definition ------------------------------
# Specialized Dataset class for the Custom data
class CustomDataset(Dataset):
    # Constructor
    def __init__(self, data: pandas.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_token_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    # Get the len of the dataset
    def __len__(self):
        return len(self.data)

    # Get the data from an specific row
    def __getitem__(self, index: int):
        # Get the row
        data_row = self.data.values[index]
        # Get the text from the news
        news_text = data_row[self.data.columns.get_loc(config.NOTICIA)]
        # Get the classification of the news
        label_names = data_row[
            self.data.columns.get_loc(config.CLASS_NAMES[0]) :
            self.data.columns.get_loc(config.CLASS_NAMES[-1]) + 1
        ]

        # Encode the news
        encoding = self.tokenizer.encode_plus(
            news_text,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Return a dictionary with the text from the news, the encoding
        # and the classification
        return dict(
            news_text=news_text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor([int(label) for label in label_names]),
        )