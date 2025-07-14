import os
import cv2
import csv
import torch
import random

from pathlib import Path
from labs import grep_files, load_json

from PIL import Image

from transformers import default_data_collator

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

def prepare_example(image_path, processor, transform):
    # load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # load metas
    json_path = Path(image_path).with_suffix(".json")
    meta = load_json(json_path)

    # words and boxes
    words, boxes = get_words_and_boxes(image, meta, use_norm=False)

    # augments
    augmented = transform(image=image, bboxes=boxes, words=words)
    image = augmented['image']
    words = augmented['words']
    boxes = augmented['bboxes']
    boxes = [to_norm_box_with_size(b, h, w) for b in boxes] 

    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    return encoding

def load_csv_targets(csv_path, encoding='utf-8'):
    labels = {}
    with open(csv_path, mode='r', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row['ID']
            value = row['target']
            labels[key] = value
    return labels

def get_words_and_boxes(image, meta, use_norm=True, score_threashold=0.3):
    words, boxes = [], []
    for it in meta['annotation']:
        if score_threashold < it['score']:
            box = it['box'] if not use_norm else to_norm_box(image, it['box'])
            boxes.append(box)
            words.append(it['text'])
    return words, boxes


def make_doc_class_mapper(json_path):
    dc = load_json(json_path)
    classes = dc['document_classes']
    
    label2id = {v: k for k, v in enumerate(classes)}
    id2label = {k: v for k, v in enumerate(classes)}

    return label2id, id2label

def to_norm_box(image:Image, box):
    w, h = image.width, image.height
    return to_norm_box_with_size(box, h, w)
    
def to_norm_box_with_size(box, h, w):
    x0, y0, x1, y1 = box

    x0_norm = int((x0 / w) * 1000)
    y0_norm = int((y0 / h) * 1000)
    x1_norm = int((x1 / w) * 1000)
    y1_norm = int((y1 / h) * 1000)

    x0_norm = max(0, min(1000, x0_norm))
    y0_norm = max(0, min(1000, y0_norm))
    x1_norm = max(0, min(1000, x1_norm))
    y1_norm = max(0, min(1000, y1_norm))

    return (x0_norm, y0_norm, x1_norm, y1_norm)


def split_ds(paths, train_ratio, valid_ratio, trial_ratio, seed):
    if (train_ratio + valid_ratio) > 1.0:
        raise ValueError("비율 합 에러")
    
    items = paths[:]
    random.seed(seed)
    random.shuffle(items)
    
    n = len(items)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    train_set = items[:train_end]
    valid_set = items[train_end:valid_end]
    trial_set = items[valid_end:]
    
    return train_set, valid_set, trial_set 



class DocsyDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform):
        self.labels = labels
        self.processor = processor
        self.transform = transform
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        encoding = prepare_example(image_path, self.processor, self.transform)
        target = int(self.labels[os.path.basename(image_path)])

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "bbox": encoding["bbox"].squeeze(0),
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": torch.tensor(target, dtype=torch.long)
        }

class DocsyDataModule(LightningDataModule):
    def __init__(
        self,
        ds_dir_path,
        targets_csv_path,
        processor,
        augmentor,
        batch_size,
        num_workers,
        train_ratio,
        valid_ratio,
        trial_ratio,
        seed=250630
    ):
        super().__init__()
        image_paths = grep_files(ds_dir_path, exts=['jpg'])
        train_paths, valid_paths, trial_paths = split_ds(image_paths,
                                                         train_ratio=train_ratio, 
                                                         valid_ratio=valid_ratio, 
                                                         trial_ratio=trial_ratio, 
                                                         seed=seed)
        self.train_paths = train_paths
        self.valid_paths = valid_paths
        self.trial_paths = trial_paths
        self.targets = load_csv_targets(targets_csv_path)
        self.augmentor = augmentor
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        if stage == "fit":
            self.train_ds = DocsyDataset(self.train_paths, 
                                         self.targets, 
                                         self.processor,
                                         self.augmentor.make_transforms())
            self.valid_ds = DocsyDataset(self.valid_paths, 
                                         self.targets, 
                                         self.processor,
                                         self.augmentor.make_transforms())
        if stage == "test" or stage is None:
            self.trial_ds = DocsyDataset(self.trial_paths, self.targets, self.processor)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=default_data_collator
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator 
        )

    def test_dataloader(self):
        return DataLoader(
            self.trial_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=default_data_collator 
        )
