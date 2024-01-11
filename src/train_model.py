import os
import random
import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from models.model import BERTClass
from utils.logger import get_logger

logger = get_logger(__name__)

# Set hyperparameters
TRAIN_BATCH_SIZE = 128
VALID_BATCH_SIZE = 64
LEARNING_RATE = 1e-05
EPOCHS = 3

config = {"train_batch_size": TRAIN_BATCH_SIZE, "valid_batch_size": VALID_BATCH_SIZE, "epochs": EPOCHS, "lr": LEARNING_RATE}
wandb.init(project="dtu-mlops", config=config)

# Set a random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Constants and parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
train_set = torch.load("data/processed/train_set.pt")
val_set = torch.load("data/processed/val_set.pt")

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

# Initialize the model, loss function, and optimizer
model = BERTClass().to(device)
wandb.watch(model, log_freq=100)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss = 0
    for _, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'Epoch {epoch} - Training'):
        ids, mask, token_type_ids, targets = [d.to(device) for d in data]

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    # Validation
    model.eval()
    total_val_loss = 0
    fin_val_targets, fin_val_outputs = [], []
    for _, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), desc=f'Epoch {epoch} - Validation'):
        ids, mask, token_type_ids, targets = [d.to(device) for d in data]

        outputs = model(ids, mask, token_type_ids)

        val_loss = loss_fn(outputs, targets)
        total_val_loss += val_loss.item()

        fin_val_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    average_val_loss = total_val_loss / len(val_loader)
    val_losses.append(average_val_loss)

    # Metrics and logging
    val_outputs = np.array(fin_val_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_val_targets, val_outputs)
    f1_score_micro = metrics.f1_score(fin_val_targets, val_outputs, average="micro")
    f1_score_macro = metrics.f1_score(fin_val_targets, val_outputs, average="macro")

    logger.info(f"Epoch {epoch}:")
    logger.info(f"  Training Loss = {average_loss}")
    logger.info(f"  Validation Loss = {average_val_loss}")

    wandb.log({"Training Loss": average_loss, "Validation Loss": average_val_loss,
               "Accuracy Score": accuracy, "F1 Score (Micro)": f1_score_micro, "F1 Score (Macro)": f1_score_macro})

# Save the model
model_path = "models/fine_tuned"
os.makedirs(model_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(model_path, "bert_model.pth"))
