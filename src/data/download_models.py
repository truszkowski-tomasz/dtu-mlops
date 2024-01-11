""" Download and save Hugging Face model locally. """


import os
import argparse
from transformers import AutoTokenizer, AutoModel
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
src_path = os.path.join(project_root, "src")
sys.path.append(src_path)

from utils.logger import get_logger

logger = get_logger(__name__)


def download_and_save_model(model_name, save_directory):
    if os.path.exists(save_directory):
        logger.info(f"Model '{model_name}' already exists in '{save_directory}'. Skipping download.")
        return

    logger.info(f"Downloading and saving Hugging Face model '{model_name}' to '{save_directory}'...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    logger.info(f"Model and tokenizer saved to {save_directory}")


def main():
    parser = argparse.ArgumentParser(description="Download and save Hugging Face model locally.")
    parser.add_argument(
        "--model_name", type=str, default="bert-base-uncased", help="Name of the Hugging Face model to download."
    )

    args = parser.parse_args()

    save_directory = os.path.join("models", args.model_name)

    download_and_save_model(args.model_name, save_directory)


if __name__ == "__main__":
    main()
