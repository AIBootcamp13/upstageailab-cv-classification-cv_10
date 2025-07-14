import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR

from transformers import LayoutLMv3ForSequenceClassification as LyLmv3, LayoutLMv3Processor

class DiceLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_true = F.one_hot(y_true, num_classes=y_pred.size(-1)).float()
        y_pred = F.softmax(y_pred, dim=-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
        return 1 - dicek

class Lym(pl.LightningModule):
    def __init__(self, label2id, id2label):
        super().__init__()
        num_classes = len(label2id)
        
        model = LyLmv3.from_pretrained("microsoft/layoutlmv3-base", num_labels=num_classes)
        self.model = model
        self.model.train()
        self.model.config.label2id = label2id
        self.model.config.id2label = id2label

        self.loss_fn = DiceLoss()
        self.dice_weight = 0.4

        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "per-class-accuracy" : MulticlassAccuracy(num_classes=num_classes, average=None),
            "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
            "F1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }

        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.valid_metrics = MetricCollection(metrics, prefix="valid_")

    def forward(self, input_ids, attention_mask, bbox, pixel_values, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            labels=labels
        )

    def feed(self, batch):
        return self(
            batch["input_ids"],
            batch["attention_mask"],
            batch["bbox"],
            batch["pixel_values"],
            batch["labels"]
        )
    