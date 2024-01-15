import torch
from transformers import BertTokenizer
from train_model import train_and_get_model

def prediction_model(input_text, model, device):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model.to(device)
    model.eval()

    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=True,  # Include token type ids
        truncation=True,
        return_tensors='pt'
    )

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)  # Get token type ids

    with torch.no_grad():
        outputs = model(ids, mask, token_type_ids)  # Include token type ids in the call
        probabilities = torch.sigmoid(outputs)
        prediction = probabilities[0][0].item() > 0.5

    return prediction


# Example usage
train_set = torch.load("data/processed/train_set.pt")
val_set = torch.load("data/processed/val_set.pt")
model = train_and_get_model(train_set, val_set)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_text = "Your news article text here"

is_real = prediction_model(input_text, model, device)
print("The news is real" if is_real else "The news is fake")
