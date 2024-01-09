""" Download and save Hugging Face model locally. """


import os
import argparse
from transformers import AutoTokenizer, AutoModel


def download_and_save_model(model_name, save_directory):
    # Download
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Save locally
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    print(f"Model and tokenizer saved to {save_directory}")


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
