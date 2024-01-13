import os
import random
import numpy as np
import pandas as pd
import torch
import wandb
from pytorch_lightning import Trainer
from sklearn import metrics
from torch.utils.data import DataLoader
from models.model import BERTLightning
from torch.profiler import profile, ProfilerActivity, record_function  # Import torch profiler

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

# Load datasets
train_set = torch.load("data/processed/train_set.pt")
val_set = torch.load("data/processed/val_set.pt")

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=2)

# Initializing the model, loss function, and optimizer
model = BERTLightning().to(device)


trainer = Trainer(max_epochs=EPOCHS, log_every_n_steps=1)

# Profiling
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function("model_training"):
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Print profiler results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
prof.export_chrome_trace("trace.json")

model_path = "models/fine_tuned"

# If the directory does not exist, create it
if not os.path.exists(model_path):
    os.mkdir(model_path)

# Save the model
torch.save(model.state_dict(), model_path+"/bert_model.pth")
