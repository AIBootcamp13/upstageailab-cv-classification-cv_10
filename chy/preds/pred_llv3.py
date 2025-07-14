import os
import cv2
import csv
import json
import shutil
import logging
import random
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image, ImageOps, ImageFont, ImageDraw

import torch
import torch.nn.functional as F

from transformers import AutoImageProcessor
from transformers import ConvNextV2ForImageClassification

default_trans = {
    "account_number":"계좌번호",
    "application_for_payment_of_pregnancy_medical_expenses": "임신/출산",
    "car_dashboard": "계기판",
    "confirmation_of_admission_and_discharge": "입퇴원 확인서",
    "diagnosis": "진단서",
    "driver_lisence": "면허증",
    "medical_bill_receipts": "의료비 영수증",
    "medical_outpatient_certificate": "진료/통원/외래 확인서",
    "national_id_card": "주민등록증",
    "passport": "여권",
    "payment_confirmation": "납입 확인서",
    "pharmaceutical_receipt": "약국 영수증",
    "prescription": "처방전",
    "resume": "이력서",
    "statement_of_opinion": "소견서",
    "vehicle_registration_certificate": "차량등록증",
    "vehicle_registration_plate": "자동차"
}

def load_json(path):
    with open(path) as f:
        return json.load(f)
      
def make_doc_class_mapper(json_path):
    dc = load_json(json_path)
    classes = dc['document_classes']
    
    label2id = {v: k for k, v in enumerate(classes)}
    id2label = {k: v for k, v in enumerate(classes)}

    return label2id, id2label

def make_id2kor(id2label):
    return {k:default_trans[v] for k, v in id2label.items()}

def write_csv_value(csv_path, preds):
    df = pd.read_csv(csv_path)
    for i, item in enumerate(preds):
        filename = os.path.basename(item[0])
        df.loc[df['ID'] == filename, 'target'] = item[1]
        
    df.to_csv(csv_path, index=False, encoding='utf-8')

def get_words_and_boxes(image, meta, score_threashold=0.3):
    words, boxes = [], []
    for it in meta['annotation']:
        if score_threashold < it['score']:
            boxes.append(it['box'])
            words.append(it['text'])
    return np.array(words), np.array(boxes)


class Predictor:
    def __init__(self, model_class, processor, ckpt_path, classes_path):
        self.model_class = model_class
        self.processor = processor
        self.ckpt_path = ckpt_path
        label2id, id2label = make_doc_class_mapper(classes_path)
        self.label2id = label2id
        self.id2label = id2label
        self.label_trans = default_trans
        self.id2kor = make_id2kor(id2label)
        self.class_names = [self.label_trans[k] for k in label2id.keys()]
        self._load_checkpoint()

    def _load_checkpoint(self):
        self.model = self.model_class.load_from_checkpoint(
            self.ckpt_path, id2label=self.id2label, label2id=self.label2id)
        self.model.eval()
        self.model.to('cuda')

    def to_example(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load metas
        json_path = Path(image_path).with_suffix(".json")
        meta = load_json(json_path)
    
        # words and boxes
        words, boxes = get_words_and_boxes(image, meta)

        example = self.processor(
            images=image,
            text=words.tolist(),
            boxes=boxes.tolist(),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt")

        return {
            "input_ids": example["input_ids"],
            "attention_mask": example["attention_mask"],
            "bbox": example["bbox"],
            "pixel_values": example["pixel_values"],
        }

    def feed(self, image_path):
        ex = self.to_example(image_path)
        
        with torch.no_grad():
            # try:
            outs = self.model(
                ex["input_ids"].cuda(),
                ex["attention_mask"].cuda(),
                ex["bbox"].long().cuda(),
                ex["pixel_values"].cuda(),
            )
            # except Exception as e:
            #     print('error in', ex, 'with', e)
            
        preds = torch.argmax(outs.logits, dim=1)
        return outs, preds

    def feed_bty(self, image_path):
        outs, preds = self.feed(image_path)
        return logits_to_prob_table(outs.logits, self.class_names)

    def test(self, image_paths, use_probs=True):
        items = []
        for p in tqdm(image_paths):
            outs, pred_ids = self.feed(p)
            if use_probs:
                v = F.softmax(outs.logits, dim=1).cpu().numpy() 
                v = [float("{:.4f}".format(p)) for p in v.tolist()[0]] 
            else:
                v = pred_ids.cpu().numpy()
            # top3_values, top3_indices = torch.topk(outs.logits, k=3, dim=1)
            # pred = (top3_values.cpu().numpy(), top3_indices.cpu().numpy())

            item = (os.path.basename(p), v)
            items.append(item)
        return items

    @staticmethod
    def make_table(model_out):
        # logits_to_prob_table(outs.logits, class_names=class_names)
        pass

def logits_to_prob_table(logits, class_names, labels=None):
    probs = F.softmax(logits, dim=1)  # (batch, num_classes)
    probs_np = probs.detach().cpu().numpy()
    batch_size, num_classes = probs_np.shape

    num_classes = probs_np.shape[1]
    if labels is not None:
        labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
    else:
        labels_np = [0] * batch_size

    records = []
    columns = ['정답'] + [f'{name}' for name in class_names]
    for i in range(batch_size):
        row = [labels_np[i]] + [ float("{:.4f}".format(p)) for p in probs_np[i]] 
        records.append(row)

    df = pd.DataFrame(records, columns=columns)
    return df
