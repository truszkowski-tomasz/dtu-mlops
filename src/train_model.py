import cProfile
import pstats
import os
import random
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics
from torch.utils.data import DataLoader
from models.model import BERTLightning

def main():
    # Set a random seed for reproducibility
    random_seed = 42
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Constants and parameters
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 1e-05
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {"train_batch_size": TRAIN_BATCH_SIZE, "valid_batch_size": VALID_BATCH_SIZE, "epochs": EPOCHS, "lr": LEARNING_RATE}
    wandb.init(project="dtu-mlops", config=config)

    # Load datasets
    train_set = torch.load("data/processed/train_set.pt")
    val_set = torch.load("data/processed/val_set.pt")

    # Create DataLoader
    train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=7)
    val_loader = DataLoader(val_set, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=7)

    # Initializing the model, loss function, and optimizer
    model = BERTLightning().to(device)

    wandb.watch(model, log_freq=100)
    logger = WandbLogger()

    trainer = Trainer(max_epochs=EPOCHS, log_every_n_steps=1, logger=logger)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    model_path = "models/fine_tuned"

    # If the directory does not exist, create it
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # Save the model
    torch.save(model.state_dict(), model_path+"/bert_model.pth")

if __name__ == "__main__":
    cProfile.run('main()', 'profiling_stats')

    # Reading the profiling stats
    p = pstats.Stats('profiling_stats')

    # Printing top 10 lines sorted by cumulative time
    print("Top functions by cumulative time:")
    p.strip_dirs().sort_stats('cumulative').print_stats(10)

    # Printing top 10 lines sorted by total time
    print("\nTop functions by total time:")
    p.strip_dirs().sort_stats('tottime').print_stats(10)
