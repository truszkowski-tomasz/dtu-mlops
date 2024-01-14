import argparse
import os
import sys

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, TensorDataset
from transformers import BertTokenizer

# For some reason, I cannot make the logger work without this workaround
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.logger import get_logger

logger = get_logger(__name__)


def preprocess_data(df, train_size, max_len):
    logger.info(f"Processing data with max_len={max_len} and train_size={train_size}")

    new_df = df[["text", "label"]].copy()
    new_df.columns = ["text", "labels"]

    # drop rows where either text or labels are missing
    new_df.dropna(inplace=True)

    # Creating the dataset and dataloader for the neural network
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    val_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    logger.info("FULL Dataset: {}".format(new_df.shape))
    logger.info("TRAIN Dataset: {}".format(train_dataset.shape))
    logger.info("VAL Dataset: {}".format(val_dataset.shape))

    # Tokenize and convert to TensorDataset
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased")
    train_data = [
        tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        for text in train_dataset["text"]
    ]
    train_input_ids = torch.cat([data["input_ids"] for data in train_data], dim=0)
    train_attention_mask = torch.cat([data["attention_mask"] for data in train_data], dim=0)
    train_token_type_ids = torch.cat([data["token_type_ids"] for data in train_data], dim=0)

    train_labels = torch.tensor(train_dataset["labels"].values, dtype=torch.float).unsqueeze(1)

    train_set = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)

    val_data = [
        tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        for text in val_dataset["text"]
    ]
    val_input_ids = torch.cat([data["input_ids"] for data in val_data], dim=0)
    val_attention_mask = torch.cat([data["attention_mask"] for data in val_data], dim=0)
    val_token_type_ids = torch.cat([data["token_type_ids"] for data in val_data], dim=0)

    val_labels = torch.tensor(val_dataset["labels"].values, dtype=torch.float).unsqueeze(1)

    val_set = TensorDataset(val_input_ids, val_attention_mask, val_token_type_ids, val_labels)

    return train_set, val_set


def save_datasets(train_set, val_set, save_path="data/processed/"):
    torch.save(train_set, f"{save_path}train_set.pt")
    torch.save(val_set, f"{save_path}val_set.pt")
    logger.info(f"Saved datasets to {save_path}")


@hydra.main(config_path="../config", config_name="default_config.yaml", version_base="1.1")
def load_and_tokenize_data(config: DictConfig) -> None:
    file_path = config.data.file_path
    df = pd.read_csv(file_path).head(config.data.subset_size) if config.data.subset else pd.read_csv(file_path)

    train_set, val_set = preprocess_data(df, config.data.train_size, config.data.max_len)
    save_datasets(train_set, val_set)


if __name__ == "__main__":
    load_and_tokenize_data()
