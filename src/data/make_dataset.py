import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch
from transformers import BertTokenizer
import argparse
import os
import sys

# For some reason, I cannot make the logger work without this workaround
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.logger import get_logger

logger = get_logger(__name__)


class FakeNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.labels = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_token_type_ids=True,
            truncation=True,
            truncation_strategy="longest_first",
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels[index], dtype=torch.float).unsqueeze(0),
        }


def preprocess_data(df, train_size, max_len):
    logger.info(f"Processing data with max_len={max_len} and train_size={train_size}")

    new_df = df[["text", "label"]].copy()
    new_df.columns = ["text", "labels"]

    # Creating the dataset and dataloader for the neural network
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    logger.info("FULL Dataset: {}".format(new_df.shape))
    logger.info("TRAIN Dataset: {}".format(train_dataset.shape))
    logger.info("TEST Dataset: {}".format(test_dataset.shape))

    # Tokenize and convert to TensorDataset
    tokenizer = BertTokenizer.from_pretrained("models/bert-base-uncased")
    train_data = [
        tokenizer.encode_plus(text, max_length=max_len, padding="max_length", return_tensors="pt", truncation=True)
        for text in train_dataset["text"]
    ]
    train_input_ids = torch.cat([data["input_ids"] for data in train_data], dim=0)
    train_attention_mask = torch.cat([data["attention_mask"] for data in train_data], dim=0)
    train_token_type_ids = torch.cat([data["token_type_ids"] for data in train_data], dim=0)

    train_labels = torch.tensor(train_dataset["labels"].values, dtype=torch.float).unsqueeze(1)

    train_set = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)

    test_data = [
        tokenizer.encode_plus(text, max_length=max_len, padding="max_length", return_tensors="pt", truncation=True)
        for text in test_dataset["text"]
    ]
    test_input_ids = torch.cat([data["input_ids"] for data in test_data], dim=0)
    test_attention_mask = torch.cat([data["attention_mask"] for data in test_data], dim=0)
    test_token_type_ids = torch.cat([data["token_type_ids"] for data in test_data], dim=0)

    test_labels = torch.tensor(test_dataset["labels"].values, dtype=torch.float).unsqueeze(1)

    test_set = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, test_labels)

    return train_set, test_set


def save_datasets(train_set, test_set, save_path="data/processed/"):
    torch.save(train_set, f"{save_path}train_set.pt")
    torch.save(test_set, f"{save_path}test_set.pt")
    logger.info(f"Saved datasets to {save_path}")


def load_and_tokenize_data(file_path, max_len, train_size, subset_size=True):
    df = pd.read_csv(file_path).head(100) if subset_size else pd.read_csv(file_path)

    train_set, test_set = preprocess_data(df, train_size, max_len)
    save_datasets(train_set, test_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and tokenize for fake news classification.")
    parser.add_argument("--file_path", default="data/raw/WELFake_Dataset.csv", type=str, help="Path to the CSV file.")
    parser.add_argument("--max_len", default=200, type=int, help="Maximum length for tokenization.")
    parser.add_argument("--train_size", default=0.8, type=float, help="Fraction of data used for training.")

    args = parser.parse_args()
    logger.info(
        f"Script started with arguments: file_path={args.file_path}, max_len={args.max_len}, train_size={args.train_size}"
    )
    load_and_tokenize_data(args.file_path, args.max_len, args.train_size)
