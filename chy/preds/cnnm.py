import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers import AutoImageProcessor
from transformers import ConvNextV2ForImageClassification

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.classification import MulticlassAccuracy
from torch.optim.lr_scheduler import ReduceLROnPlateau, ConstantLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR

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

class CNN(pl.LightningModule):
    def __init__(self, label2id, id2label):
        super().__init__()
        num_classes = len(label2id)
        self.model = ConvNextV2ForImageClassification.from_pretrained('facebook/convnextv2-large-22k-384', num_labels=num_classes, ignore_mismatched_sizes=True)
        self.model.train()
        self.model.config.label2id = label2id
        self.model.config.id2label = id2label

        self.loss_fn = DiceLoss()
        self.dice_weight = 0.4

        metrics = {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            "per-class-accuracy" : MulticlassAccuracy(num_classes=num_classes, average=None),
            "roc_auc": AUROC(task="multiclass", num_classes=num_classes),
            "precision": Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall": Recall(task="multiclass", num_classes=num_classes, average="macro"),
            "F1": F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }

        self.train_metrics = MetricCollection(metrics, prefix="train_")
        self.valid_metrics = MetricCollection(metrics, prefix="valid_")

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=3)

        scheduler_delay = ConstantLR(optimizer, factor=1.0, total_iters=30)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.01, total_iters=10)
        total_training_steps = 1000 # 총 학습 스텝 수
        scheduler_main = CosineAnnealingLR(optimizer, T_max=total_training_steps - 40, eta_min=1e-6)
        final_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_delay, scheduler_warmup, scheduler_main],
            milestones=[40, 50]
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": final_scheduler,
                "interval": "step",
                "monitor": "valid_loss"
            }
        }
     
    def feed(self, batch):
        return self(batch["pixel_values"], batch["labels"])