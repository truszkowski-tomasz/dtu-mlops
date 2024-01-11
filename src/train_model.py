import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from sklearn import metrics
from torch.utils.data import DataLoader
from models.model import BERTLightning

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

train_set = torch.load("data/processed/train_set.pt")
val_set = torch.load("data/processed/val_set.pt")

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=7)
val_loader = DataLoader(val_set, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=7)

# Initializing the model, loss function, and optimizer
model = BERTLightning()
model.to(device)

wandb.watch(model, log_freq=100)
logger = WandbLogger()

trainer = Trainer(max_epochs=EPOCHS, log_every_n_steps=1, logger=logger)

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


# Plotting loss changes
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label="Training Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# If the directory does not exist, create it
if not os.path.exists("models/fine_tuned"):
    os.mkdir("models/fine_tuned")

# Save the model
torch.save(model.state_dict(), "models/fine_tuned/bert_model.pth")
