import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
    Main function to process the dataset.

    Returns:
        None
    """
    # TODO: raw_path should be in a config file
    raw_path = 'data/raw'
    # TODO: processed_path should be in a config file
    processed_path = 'data/processed'
    process_data(raw_path, processed_path)

if __name__ == '__main__':
    main()