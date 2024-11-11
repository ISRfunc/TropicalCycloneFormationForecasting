import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from torchmetrics import Accuracy, F1Score, Precision, Recall
from pytorch_lightning.utilities.grads import grad_norm
import torch.nn.functional as F
from .losses import FocalLoss


class ModelWrapper(L.LightningModule):
    def __init__(self, model, learning_rate: float=0.001, decision_boundary: float=0.85, pos_weight: float=1):
        super().__init__()

        self.learning_rate = learning_rate

        self.model = model 

        self.example_input_array3D = torch.randn(1, 1, 465, 33, 33)

        self.decision_boundary = decision_boundary

        self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([pos_weight]))

        self.f1 = F1Score(task="binary", threshold=self.decision_boundary)
        self.accuracy = Accuracy(task="binary", threshold=self.decision_boundary)
        self.precision = Precision(task="binary", threshold=self.decision_boundary)
        self.recall = Recall(task="binary", threshold=self.decision_boundary)



    def forward(self, X): 
        return self.model(X)
    
    def training_step(self, batch, batch_idx): 
        X, labels = batch

        logits = self(X).squeeze(1)
        probs = torch.sigmoid(logits)

        acc = self.accuracy(probs, labels)
        f1 = self.f1(probs, labels)
        pre = self.precision(probs, labels)
        rec = self.recall(probs, labels)

        loss = self.loss_fn(probs, labels)

        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1, prog_bar=True)
        self.log("train_precision", pre, prog_bar=True)
        self.log("train_recall", rec, prog_bar=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, acc, f1, pre, rec = self._shared_eval(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("val_recall", rec, prog_bar=True)
        self.log("val_precision", pre, prog_bar=True)

        return {"val_loss": loss, "val_accuracy": acc, "val_f1": f1, "val_precision": pre, "val_recall": rec}

    def test_step(self, batch, batch_idx):
        loss, acc, f1, pre, rec = self._shared_eval(batch)
        self.log_dict({"test_loss": loss, "test_acc": acc, "test_f1": f1, "test_precision": pre, "test_recall": rec})

    def _shared_eval(self, batch):
        X, labels = batch 

        logits = self(X).squeeze(1)
        probs = torch.sigmoid(logits)

        acc = self.accuracy(probs, labels)
        f1 = self.f1(probs, labels)
        pre = self.precision(probs, labels)
        rec = self.recall(probs, labels)


        loss = self.loss_fn(probs, labels)

        return loss, acc, f1, pre, rec

    def on_before_optimizer_step(self, optimizer):
        # Compute the L2 norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)


    def configure_optimizers(self):

        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(
            params, lr=self.learning_rate, weight_decay=0.0001
        )
        scheduler = StepLR(
            optimizer, step_size=5, gamma=0.8
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
