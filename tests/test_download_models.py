import os
import pytest
from unittest.mock import patch
from src.data.download_models import download_and_save_model

@pytest.fixture
def save_directory(tmpdir):
    return str(tmpdir)

def test_download_and_save_model_existing_directory(save_directory, caplog):
    # Create a dummy file in the save directory to simulate an existing model
    os.makedirs(save_directory, exist_ok=True)
    with open(os.path.join(save_directory, "dummy_model.bin"), "w") as f:
        f.write("dummy content")

    # Call the function with an existing save directory
    download_and_save_model("dummy_model", save_directory)

    # Check that the function logs a  and returns early
    assert "Model 'dummy_model' already exists" in caplog.text
    assert not os.path.exists(os.path.join(save_directory, "dummy_model"))

    # Remove the dummy file and model
    os.remove(os.path.join(save_directory, "dummy_model.bin"))
    os.rmdir(save_directory)
