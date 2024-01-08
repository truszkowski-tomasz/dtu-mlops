# train_model.py
import numpy as np
import pandas as pd
from sklearn import metrics
from torch.utils.data import DataLoader
from data.make_dataset_2 import CustomDataset
from models.model import BERTClass
from transformers import BertTokenizer
import torch 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
df = pd.read_csv("data/raw/WELFake_Dataset.csv").head(10)

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = BERTClass()
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 5000 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask, token_type_ids)

            # Convert outputs to binary predictions
            predictions = torch.sigmoid(outputs).cpu().detach().numpy() >= 0.5

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(predictions.tolist())

    outputs = np.array(fin_outputs)
    accuracy = metrics.accuracy_score(fin_targets, outputs)
    f1_score_micro = metrics.f1_score(fin_targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(fin_targets, outputs, average='macro')
    print(f"Validation Accuracy Score = {accuracy}")
    print(f"Validation F1 Score (Micro) = {f1_score_micro}")
    print(f"Validation F1 Score (Macro) = {f1_score_macro}")


for epoch in range(EPOCHS):
    train(epoch)

torch.save(model.state_dict(), 'models/fine_tuned/bert.pth')