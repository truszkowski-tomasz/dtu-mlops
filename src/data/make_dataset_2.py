import pandas as pd
from torch.utils.data import Dataset, TensorDataset
import torch
from transformers import BertTokenizer

class FakeNewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text 
        self.labels = self.data.labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            truncation_strategy='longest_first'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.float).unsqueeze(0)
        }
    
def preprocess_data(df, train_size, max_len):
    new_df = df[['text', 'label']].copy()
    new_df.columns = ['text', 'labels']

    # Creating the dataset and dataloader for the neural network
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Tokenize and convert to TensorDataset
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')
    train_data = [tokenizer.encode_plus(text, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True) for text in train_dataset['text']]
    train_input_ids = torch.cat([data['input_ids'] for data in train_data], dim=0)
    train_attention_mask = torch.cat([data['attention_mask'] for data in train_data], dim=0)
    train_token_type_ids = torch.cat([data['token_type_ids'] for data in train_data], dim=0)
    
    train_labels = torch.tensor(train_dataset['labels'].values, dtype=torch.float).unsqueeze(1)

    train_set = TensorDataset(train_input_ids, train_attention_mask, train_token_type_ids, train_labels)

    test_data = [tokenizer.encode_plus(text, max_length=max_len, padding='max_length', return_tensors='pt', truncation=True) for text in test_dataset['text']]
    test_input_ids = torch.cat([data['input_ids'] for data in test_data], dim=0)
    test_attention_mask = torch.cat([data['attention_mask'] for data in test_data], dim=0)
    test_token_type_ids = torch.cat([data['token_type_ids'] for data in test_data], dim=0)

    test_labels = torch.tensor(test_dataset['labels'].values, dtype=torch.float).unsqueeze(1)

    test_set = TensorDataset(test_input_ids, test_attention_mask, test_token_type_ids, test_labels)

    return train_set, test_set

def save_datasets(train_set, test_set, save_path="data/processed/"):
    torch.save(train_set, f"{save_path}train_set.pt")
    torch.save(test_set, f"{save_path}test_set.pt")
    print("Files have been saved.")

def load_and_tokenize_data(file_path, max_len, train_size):
    df = pd.read_csv(file_path).head(100)
    train_set, test_set = preprocess_data(df, train_size, max_len)
    save_datasets(train_set, test_set)

if __name__ == "__main__":
    load_and_tokenize_data("data/raw/WELFake_Dataset.csv", 200, 0.8)
