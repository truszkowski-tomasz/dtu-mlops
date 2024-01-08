import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from data.make_dataset_2 import FakeNewsDataset, load_and_tokenize_data
from models.model import BERTClass
import matplotlib.pyplot as plt
import random
from torch import cuda

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

train_set = torch.load("data/processed/train_set.pt")
test_set = torch.load("data/processed/test_set.pt")

# Create DataLoader
train_loader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=0)

# Initializing the model, loss function, and optimizer
model = BERTClass()
model.to(device)

loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

# Training loop
train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for _, data in enumerate(train_loader, 0):
        ids, mask, token_type_ids, targets = data
        ids, mask, token_type_ids, targets = ids.to(device), mask.to(device), token_type_ids.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    train_losses.append(average_loss)

    model.eval()
    total_val_loss = 0
    fin_val_targets = []
    fin_val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids, mask, token_type_ids, targets = data
            ids, mask, token_type_ids, targets = ids.to(device), mask.to(device), token_type_ids.to(device), targets.to(device)

            outputs = model(ids, mask, token_type_ids)

            val_loss = loss_fn(outputs, targets)
            total_val_loss += val_loss.item()

            fin_val_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    average_val_loss = total_val_loss / len(test_loader)
    val_losses.append(average_val_loss)

    val_outputs = np.array(fin_val_outputs) >= 0.5
    accuracy = metrics.accuracy_score(fin_val_targets, val_outputs)
    f1_score_micro = metrics.f1_score(fin_val_targets, val_outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_val_targets, val_outputs, average='macro')

    print(f"Epoch {epoch}:")
    print(f"  Training Loss = {average_loss}")
    print(f"  Validation Loss = {average_val_loss}")
    print(f"  Accuracy Score = {accuracy}")
    print(f"  F1 Score (Micro) = {f1_score_micro}")
    print(f"  F1 Score (Macro) = {f1_score_macro}")

# Plotting loss changes
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'models/fine_tuned/bert_model.pth')