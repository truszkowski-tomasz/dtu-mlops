import numpy as np
import pandas as pd
from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from data.make_dataset_2 import FakeNewsDataset
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
MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')


# Loading the dataset
df = pd.read_csv("data/raw/WELFake_Dataset.csv").head(10)
new_df = df[['text', 'label']].copy()
new_df.columns = ['text', 'labels']

# Creating the dataset and dataloader for the neural network
train_size = 0.8
train_dataset=new_df.sample(frac=train_size,random_state=200)
test_dataset=new_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)


print("FULL Dataset: {}".format(new_df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = FakeNewsDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = FakeNewsDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

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
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['labels'].to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(training_loader)
    train_losses.append(average_loss)

    model.eval()
    total_val_loss = 0
    fin_val_targets = []
    fin_val_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['labels'].to(device, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            val_loss = loss_fn(outputs, targets)
            total_val_loss += val_loss.item()

            fin_val_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    average_val_loss = total_val_loss / len(testing_loader)
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
