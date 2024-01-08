import pandas as pd
from torch.utils.data import Dataset
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
    
def load_and_tokenize_data(file_path, max_len, train_size):
    tokenizer = BertTokenizer.from_pretrained('models/bert-base-uncased')

    df = pd.read_csv(file_path).head(10)
    new_df = df[['text', 'label']].copy()
    new_df.columns = ['text', 'labels']

    # Creating the dataset and dataloader for the neural network
    train_dataset = new_df.sample(frac=train_size, random_state=200)
    test_dataset = new_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(new_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = FakeNewsDataset(train_dataset, tokenizer, max_len)
    testing_set = FakeNewsDataset(test_dataset, tokenizer, max_len)

    return training_set, testing_set
