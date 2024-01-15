import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.models.model import BERTLightning

from torch.profiler import profile, ProfilerActivity


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
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
    model = BERTLightning(config=config).to(device)

    wandb.watch(model, log_freq=100)
    logger = WandbLogger()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        trainer = Trainer(max_epochs=config.train.epochs, log_every_n_steps=1, logger=logger)

        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
    prof.export_chrome_trace("trace.json")

    # If the directory does not exist, create it
    if not os.path.exists(config.fine_tuned_path):
        os.mkdir(config.fine_tuned_path)

    # Save the model
    torch.save(model.state_dict(), config.fine_tuned_path + "/bert_model.pth")


if __name__ == "__main__":
    train()
