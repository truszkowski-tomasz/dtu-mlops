import os
import random
import numpy as np
import pandas as pd
import torch
import wandb
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics
from torch.utils.data import DataLoader
from models.model import BERTLightning


@hydra.main(config_path="config", config_name="default_config.yaml", version_base='1.1')
def train(config: DictConfig) -> None:

    # Set a random seed for reproducibility
    torch.manual_seed(config.train.random_seed)
    np.random.seed(config.train.random_seed)
    random.seed(config.train.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project="dtu-mlops")

    # Load datasets
    train_set = torch.load("data/processed/train_set.pt")
    val_set = torch.load("data/processed/val_set.pt")

    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size_train, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size_val, shuffle=False, num_workers=7)

    # Initializing the model, loss function, and optimizer
    model = BERTLightning().to(device)

    wandb.watch(model, log_freq=100)
    logger = WandbLogger()

    trainer = Trainer(max_epochs=config.train.epochs, log_every_n_steps=1, logger=logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = "models/fine_tuned"

    # If the directory does not exist, create it
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Save the model
    torch.save(model.state_dict(), model_path+"/bert_model.pth")

if __name__ == "__main__":
    train()
