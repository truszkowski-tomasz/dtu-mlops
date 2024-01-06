import os
import zipfile
from typing import Union
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split

def download_dataset(dataset_name: str, destination_path: str) -> None:
    """
    Download the Kaggle dataset using the Kaggle API.

    Parameters:
        dataset_name (str): The name of the Kaggle dataset (e.g., 'username/dataset').
        destination_path (str): The local path where the dataset files will be downloaded.

    Returns:
        None
    """
    api = KaggleApi()
    # TODO: Add instructions to get the Kaggle API token in the README
    api.authenticate()

    # Download dataset
    api.dataset_download_files(dataset_name, path=destination_path)

    with zipfile.ZipFile(os.path.join(destination_path, dataset_name.split('/')[1] + '.zip'), 'r') as zip_ref:
        zip_ref.extractall(destination_path)

def process_data(input_path: str, output_path: str) -> None:
    """
    Load, process, and save the dataset.

    Parameters:
        input_path (str): The path where the raw dataset is located.
        output_path (str): The path where the processed dataset will be saved.

    Returns:
        None
    """
    # Load the CSV file
    df = pd.read_csv(os.path.join(input_path, 'WELFake_Dataset.csv'))

    # remove the first column (index)
    df.drop(df.columns[0], axis=1, inplace=True)

    # TODO: Data processing logic
    # Embeddings stuff

    # Split into train and validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the train and validation data
    train_path = os.path.join(output_path, 'train_data.csv')
    val_path = os.path.join(output_path, 'val_data.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

# TODO: add hydra
def main() -> None:
    """
    Main function to download and process the Kaggle dataset.

    Returns:
        None
    """
    # TODO: dataset_name should be in a config file
    # TODO: raw_path should be in a config file
    dataset_name = 'saurabhshahane/fake-news-classification'
    raw_path = 'data/raw'

    download_dataset(dataset_name, raw_path)

    # TODO: processed_path should be in a config file
    processed_path = 'data/processed'
    process_data(raw_path, processed_path)

if __name__ == '__main__':
    main()