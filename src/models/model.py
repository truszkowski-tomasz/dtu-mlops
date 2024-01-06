import torch
from torch import nn
from transformers import AutoModel, AutoConfig

class MyBaseTransformerModel(nn.Module):
    def __init__(self, model_name='models/distilbert-base-uncased'):
        super(MyBaseTransformerModel, self).__init__()
        self.num_labels = 2
        config = AutoConfig.from_pretrained(model_name, num_labels=self.num_labels)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(last_hidden_states)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss

        return logits