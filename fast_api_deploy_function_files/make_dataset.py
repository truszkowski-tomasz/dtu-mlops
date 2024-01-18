import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import TensorDataset
from transformers import BertTokenizer

from src.utils.logger import get_logger

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

    train_set = tokenize_and_convert(max_len, train_dataset)

    val_set = tokenize_and_convert(max_len, val_dataset)

    return train_set, val_set

def tokenize_and_convert(max_len, dataset):
    # Tokenize and convert to TensorDataset
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased")

    tokenized_data = [
        tokenizer.encode_plus(
            text,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        for text in dataset["text"]
    ]

    input_ids = torch.cat([data["input_ids"] for data in tokenized_data], dim=0)
    attention_mask = torch.cat([data["attention_mask"] for data in tokenized_data], dim=0)
    token_type_ids = torch.cat([data["token_type_ids"] for data in tokenized_data], dim=0)

    if "labels" in dataset.columns:
        labels = torch.tensor(dataset["labels"].values, dtype=torch.float).unsqueeze(1)
        return TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    else:
        return TensorDataset(input_ids, attention_mask, token_type_ids)

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
