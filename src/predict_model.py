import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig
from src.data.make_dataset import tokenize_and_convert
from src.models.model import BERTLightning
from src.utils.logger import get_logger

# from data.make_dataset import tokenize_and_convert
# from models.model import BERTLightning
# from utils.logger import get_logger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
def classify(cfg: DictConfig) -> np.ndarray:
    """
    Run prediction for a given dataframe

    Args:
    ----
        cfg: DictConfig
            Configuration object
        df: pd.DataFrame
            Dataframe containing the text to be predicted
    Returns:
    -------
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model
    """

    logger = get_logger(__name__)

    # Load the model from the specific checkpoint
    model = BERTLightning.load_from_checkpoint(
        cfg.model.fine_tuned_path + "/bert_model.ckpt", config=cfg
    )

    texts = cfg.predict.texts

    # Create a dataframe with the expected columns
    df = pd.DataFrame(
        {
            # title should be a list of empty strings, as long as the number of texts
            "title": [""] * len(texts),
            "text": texts,
        }
    )

    # Preprocess the data
    tensorDataset = tokenize_and_convert(cfg.data.max_len, df)

    dataloader = DataLoader(tensorDataset, batch_size=1, num_workers=7)

    trainer = Trainer()

    # Run prediction
    predictions = trainer.predict(model, dataloader)

    # Turn the predictions from a list of tensors to a flat numpy array
    predictions = np.concatenate(predictions).ravel()

    logger.info(f"Predictions: {predictions}")

    return predictions


if __name__ == "__main__":
    classify()
