import os

import torch
from torch.utils.data import TensorDataset

from src.data.make_dataset import save_datasets


def test_save_datasets(tmpdir):
    # Create a temporary directory for testing
    tmp_dir = tmpdir.mkdir("test_data")

    # Create dummy datasets
    train_set = TensorDataset(torch.rand(10, 5), torch.rand(10, 5))
    val_set = TensorDataset(torch.rand(5, 5), torch.rand(5, 5))

    # Save datasets using the function
    save_path = str(tmp_dir) + "/"
    save_datasets(train_set, val_set, save_path)

    # Check if the datasets are saved
    assert os.path.exists(os.path.join(save_path, "train_set.pt"))
    assert os.path.exists(os.path.join(save_path, "val_set.pt"))
