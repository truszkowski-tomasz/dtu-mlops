import torch
import hydra
import pandas as pd
from omegaconf import DictConfig
from data.make_dataset import tokenize_and_convert
from models.model import BERTLightning
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from src.utils.logger import get_logger

@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
def predict(config: DictConfig) -> None:
    """
    Run prediction for a given dataframe

    Args:
    ----
        df: pd.DataFrame
            Dataframe containing the text to be predicted
        config: DictConfig
            Configuration object
    Returns:
    -------
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """

    logger = get_logger(__name__)

    # Load the model from the specific checkpoint
    model = BERTLightning.load_from_checkpoint(config.model.fine_tuned_path + "/bert_model.ckpt", config=config)

    model.eval()

    # Create a dataframe table with 2 samples, each with title and text
    df = pd.DataFrame(
        {
            "title": [
                "The best movie I have ever seen",
                "The worst movie I have ever seen",
            ],
            "text": [
                "This movie was great. The acting was great and the plot was great!",
                "This movie was bad. The acting was bad and the plot was bad!",
            ],
        })

    print(df)
    print(df.columns)

    # Preprocess the data 
    tensorDataset = tokenize_and_convert(config.data.max_len, df)

    dataloader = DataLoader(tensorDataset, batch_size=10, num_workers=7)

    trainer = Trainer()

    # Run prediction
    predictions = trainer.predict(model, dataloader)

    logger.info(f"Predictions: {predictions}")

    return predictions

if __name__ == "__main__":
    predictions = predict()