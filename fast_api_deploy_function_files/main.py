import pandas as pd
import numpy as np
from flask import escape
from model import BERTLightning
from logger import get_logger
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from make_dataset import tokenize_and_convert

from google.cloud import storage
import pickle

BUCKET_NAME = "fake-news-bucket-10"
MODEL_FILE = "bert_model.ckpt"

def classify(request):
    """
    HTTP Cloud Function to run prediction for a given dataframe.

    Args:
    ----
        request: flask.Request
            HTTP request object containing configuration and data.
    """

    request_json = request.get_json(silent=True)
    request_args = request.args

    if request_json and 'texts' in request_json:
        texts = request_json['texts']
    elif request_args and 'texts' in request_args:
        texts = request_args['texts']
    else:
        return 'No texts provided for classification.', 400

    # Placeholder for actual configuration
    cfg = {
        "data": {
            "max_len": 128
        }
    }

    logger = get_logger(__name__)

    # Load the model from Google Cloud Storage
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob = bucket.get_blob(MODEL_FILE)
    my_model = pickle.loads(blob.download_as_string())

    # Create a dataframe
    df = pd.DataFrame({"title": [""] * len(texts), "text": texts})

    # Preprocess the data
    tensorDataset = tokenize_and_convert(cfg["data"]["max_len"], df)

    dataloader = DataLoader(tensorDataset, batch_size=1, num_workers=7)

    trainer = Trainer()

    # Run prediction
    predictions = trainer.predict(my_model, dataloader)

    # Turn the predictions into a flat numpy array
    predictions = np.concatenate(predictions).ravel()

    logger.info(f"Predictions: {predictions}")

    return {"predictions": predictions.tolist()}