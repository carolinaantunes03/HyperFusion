import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix


class PlModelWrapObstetrics(pl.LightningModule):
    def __init__(self, model, batch_size, optimizer, loss, class_names):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.batch_size = batch_size

        self.lr = optimizer["lr"]
        self.weight_decay = optimizer["weight_decay"]

        self.class_weights = loss["class_weights"]
        if isinstance(self.class_weights, list):
            self.class_weights = torch.tensor(self.class_weights)

        self.class_names = class_names

        self.best_val_balanced_acc = 0
        self.validation_outputs = []  # store outputs for epoch

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        imgs = batch["images"]
        num = batch["numerical"]
        cat = batch["categorical"]
        y = batch["label"] 
        tabular = (num, cat)
        logits = self((imgs, tabular))
        return logits, y

    def training_step(self, batch, batch_idx):
        logits, y = self._step(batch)
        loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        preds = torch.argmax(logits, dim=1)

        acc = torchmetrics.functional.accuracy(preds, y, task="binary")
        bal_acc = torchmetrics.functional.accuracy(preds, y, task="binary", average="macro")
        auc = torchmetrics.functional.auroc(logits.softmax(dim=-1)[:,1], y, task="binary")

        self.log("train/loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)
        self.log("train/bal_acc", bal_acc, on_epoch=True)
        self.log("train/auc", auc, on_epoch=True)

        return loss

    def _eval_step(self, batch, stage="val"):
        logits, y = self._step(batch)
        loss = F.cross_entropy(logits, y, weight=self.class_weights.to(self.device))
        preds = torch.argmax(logits, dim=1)

        self.log(f"{stage}/loss", loss, on_epoch=True, batch_size=self.batch_size)
        self.log(f"{stage}/acc", torchmetrics.functional.accuracy(preds, y, task="binary"),
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}/bal_acc", torchmetrics.functional.accuracy(preds, y, task="binary", average="macro"),
                 on_epoch=True)
        self.log(f"{stage}/auc", torchmetrics.functional.auroc(logits.softmax(dim=-1)[:,1], y, task="binary"),
                 on_epoch=True)

        return {"logits": logits, "y": y}

    # validation
    def validation_step(self, batch, batch_idx):
        output = self._eval_step(batch, "val")
        self.validation_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        # Aggregate outputs
        if len(self.validation_outputs) == 0:
            return

        logits = torch.cat([x["logits"] for x in self.validation_outputs])
        y = torch.cat([x["y"] for x in self.validation_outputs])

        preds = torch.argmax(logits, dim=1)
        bal_acc = torchmetrics.functional.accuracy(preds, y, task="binary", average="macro")

        if bal_acc > self.best_val_balanced_acc:
            self.best_val_balanced_acc = bal_acc

        self.log("val/best_bal_acc", self.best_val_balanced_acc, prog_bar=True)
        self.validation_outputs = []  # clear for next epoch

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    




class PlModelWrapEnsemble(pl.LightningModule):
    def __init__(self, model, class_names=None):
        super().__init__()
        self.model = model
        self.class_names = class_names
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        # Extract from dict
        numerical = batch["numerical"]
        categorical = batch["categorical"]
        img = batch["images"]
        image_valid = batch["image_valid_num"]  # if needed
        y = batch["label"]

        # Combine tabular features (adjust if needed)
        tabular = (numerical, categorical)
        logits = self.model((img, tabular))

        #probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.test_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": y.detach().cpu(),
            "probs": torch.softmax(logits, dim=1).detach().cpu()
            })
    
    def on_test_epoch_end(self):
    

        preds = torch.cat([x["preds"] for x in self.test_outputs])
        probs = torch.cat([x["probs"] for x in self.test_outputs])
        targets = torch.cat([x["targets"] for x in self.test_outputs])

        preds = preds.numpy()
        probs = probs.numpy()
        targets = targets.numpy()

        # ---- Metrics ----
        f1_macro = f1_score(targets, preds, average="macro")
        f1_weighted = f1_score(targets, preds, average="weighted")

        if probs.shape[1] == 2:
            roc_auc = roc_auc_score(targets, probs[:, 1])
        else:
            roc_auc = roc_auc_score(targets, probs, multi_class="ovr")

        cm = confusion_matrix(targets, preds)

        print("\n📊 TEST METRICS")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("Confusion Matrix:\n", cm)

        # clean memory
        self.test_outputs.clear()