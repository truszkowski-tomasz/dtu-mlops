import torch
from torch import nn
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from models.model import MyBaseTransformerModel
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model_name='models/distilbert-base-uncased', lr=1e-3, batch_size=16, num_epochs=3):
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Number of Epochs: {num_epochs}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    ### TO BE DELETED

    import pandas as pd

    # Load the CSV file
    csv_file_path = "data/raw/WELFake_Dataset.csv"
    df = pd.read_csv(csv_file_path)

    # Take the first 100 rows
    df_subset = df.head(100)

    # Extract texts and labels
    texts = df_subset['text'].tolist()
    labels = df_subset['label'].tolist()

    ### 

    # Tokenize input texts
    tokenized_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

    # Create PyTorch dataset
    dataset = TensorDataset(tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], torch.tensor(labels))

    # Split dataset into train and validation sets
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = MyBaseTransformerModel(model_name=model_name)  # Adjust num_labels as per your classification task
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    train_losses = []
    val_losses = []  # Add an array to store validation losses

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids, attention_mask, label = [tensor.to(device) for tensor in batch]

            logits = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_dataloader)
        train_losses.append(average_loss)
        print(f"Epoch {epoch} - Training Loss: {average_loss}")

        # Validation loop
        model.eval()
        total_correct = 0
        total_samples = 0
        val_loss = 0.0

        with torch.no_grad():
            for val_batch in val_dataloader:
                val_input_ids, val_attention_mask, val_label = [tensor.to(device) for tensor in val_batch]

                val_outputs = model(val_input_ids, attention_mask=val_attention_mask)

                # Squeeze the output tensor to remove the extra dimension
                val_loss += loss_fn(val_outputs.squeeze(), val_label).item()

                val_preds = torch.argmax(val_outputs, dim=1)

                total_correct += (val_preds == val_label).sum().item()
                total_samples += len(val_label)

        # Calculate average validation loss
        average_val_loss = val_loss / len(val_dataloader)
        val_losses.append(average_val_loss) 

    # Plot both training and validation losses
    plt.title('Training and Validation Losses')
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    torch.save(model, f"{model_name}_fine_tuned.pt")

if __name__ == "__main__":
    train()
