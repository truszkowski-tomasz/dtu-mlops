from typing import Mapping

import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from sklearn import metrics
from torch import nn

LOCAL_MODEL_PATH = "models/bert-base-uncased"


class BERTLightning(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = transformers.BertModel.from_pretrained(LOCAL_MODEL_PATH)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)

        self.criterium = torch.nn.BCEWithLogitsLoss()

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def training_step(self, batch, batch_idx):
        ids, mask, token_type_ids, targets = batch
        outputs = self(ids, mask, token_type_ids).squeeze(1)
        loss = self.criterium(outputs, targets.squeeze(1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(
            self,
            batch,
            batch_idx
        ) -> Mapping[str, torch.Tensor]:
        ids, mask, token_type_ids, targets = batch
        outputs = self(ids, mask, token_type_ids).squeeze(1)

        loss = self.criterium(outputs, targets.squeeze(1))
        accuracy = metrics.accuracy_score(targets.cpu(), torch.round(torch.sigmoid(outputs)).cpu())
        f1_score_micro = metrics.f1_score(targets.cpu(), torch.round(torch.sigmoid(outputs)).cpu(), average="micro")
        f1_score_macro = metrics.f1_score(targets.cpu(), torch.round(torch.sigmoid(outputs)).cpu(), average="macro")

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "accuracy": torch.tensor(accuracy),
            "f1_score_micro": torch.tensor(f1_score_micro),
            "f1_score_macro": torch.tensor(f1_score_macro),
        }
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer
    