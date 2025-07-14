import os
import csv
import shutil
import logging
import random
import json
import cv2
import numpy as np

from tqdm.notebook import tqdm
from dataclasses import dataclass, asdict, field
from typing import Dict, Protocol, Callable, Iterable, List

from pathlib import Path
from PIL import Image, ImageOps, ImageFont, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from torchmetrics import MetricCollection, Accuracy, F1Score, Precision, Recall, AUROC
from torchmetrics.classification import MulticlassAccuracy

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from transformers import default_data_collator

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback, TQDMProgressBar



'''
  dashboard & safetensor
'''
import wandb

def save_model_tensor(model, name):
    if not model:
        model = model.load_from_checkpoint("checkpoint.ckpt")
    state_dict = model.state_dict()
    save_file(state_dict, f"{name}.safetensors")


'''
  file processing
'''
def load_json(path):
  with open(path) as f:
    return json.load(f)

def extension(path):
    return os.path.splitext(path)[-1][1:].lower()
    
def grep_files(dir_path: str, exts=None) -> List[str]:
    if exts is None:
        exts = []
    exts = [e.lower() for e in exts]

    greps = []
    for root, dirs, files in tqdm(os.walk(dir_path)):
        for file in files:
            greps.append(os.path.join(root, file))

    if not exts:
        return greps
    return [f for f in greps if extension(f) in exts]


def pretty_json(data):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # numpy int32, float32 등도 처리
            return obj.item()
        elif obj.__class__.__name__ == 'Font':  # Font 타입은 None으로 버림
            return None
        else:
            return str(obj)  # 최후의 수단: 문자열로 덤프 가능하게

    print(json.dumps(data, indent=4, ensure_ascii=False, default=convert))

# def pretty_json(data):
#   print(json.dumps(data, indent=4, ensure_ascii=False))



'''
  Paddle OCR
'''
@dataclass
class DetectItem:
    text: str = ''
    box: List = field(default_factory=list)
    score: str = 0.0 

    def to_dict(self):
        return { 'text': self.text, 'box': self.box.tolist(), 'score': self.score }

def write_annotation(image_path: str, items: List[DetectItem]):
    data = {
        'image_path': os.path.basename(image_path),
        'annotation': [item.to_dict() for item in items]
    }

    p = Path(image_path)
    json_path = p.with_suffix('.json') 
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

class OCRProcessor:
    def __init__(self):
        from paddleocr import PPStructureV3, PaddleOCR
        # PaddleOCR(lang='korean', use_textline_orientation=False)
        # PaddleOCR(use_textline_orientation=False, text_recognition_model_name=ocr_model_name)
        ocr_model_name = 'korean_PP-OCRv5_mobile_rec'
        self.pipeline = PPStructureV3(text_recognition_model_name=ocr_model_name)

    def process_image(self, image_path):
        output = self.pipeline.predict(
            input=image_path,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False)

        if len(output) < 1:
            return []
            
        res = output[0]
        r = res['overall_ocr_res']
        texts = r['rec_texts'] 
        boxes = r['rec_boxes']
        scores = r['rec_scores']
        
        items = []
        for i in range(len(texts)):
            di = DetectItem(texts[i], boxes[i], scores[i])
            items.append(di)
            
        return items

    def draw_ocr(self, image_path, thickness=4, ratio=4.0, font_size=8):
        items = self.process_image(image_path)
        image = Image.open(image_path).convert("RGB")
        image = ImageOps.exif_transpose(image)
        
        painted = paint_ocr(image, items, thickness=thickness, ratio=ratio, font_size=font_size)
        show_img(image=painted)
        
        return painted
    

'''
 dataset
'''
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
    
def prepare_example(image_path, processor, transform):
    # load image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # correct orientation

    # load metas
    json_path = Path(image_path).with_suffix(".json")
    meta = load_json(json_path)

    # words and boxes
    words, boxes = get_words_and_boxes(image, meta)
    
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    return encoding

def split_ds(paths, train_ratio=0.6, valid_ratio=0.3, test_ratio=0.1, seed=250626):
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("비율의 합은 1이어야 합니다.")
    
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



'''
 image processing
'''
def calculate_mean_std(image_list: List[np.ndarray]):
    # 이미지 리스트를 (N, H, W, C) 형태로 스택
    images = np.stack(image_list, axis=0)
    
    # 각 채널별 평균과 표준편차 계산 (0~1 범위로 정규화)
    mean = np.mean(images, axis=(0, 1, 2)) / 255.0
    std = np.std(images, axis=(0, 1, 2)) / 255.0
    
    return tuple(mean.tolist()), tuple(std.tolist())

def show_img(path=None, image=None, figsize=(10,10)):
  if path:
    img = mpimg.imread(path)
  elif image is not None:
    img = image
  else:
    return

  print(f"shape HWC:{img.shape}") 
  plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.axis("off")
  plt.show()

def get_random_color():
  return tuple(np.random.randint(0, 256, 3).tolist())

def paint_text(image, text, box, color, ratio=1.0, font_size=10):
    # OpenCV(BGR) -> PIL(RGB) 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_img)

    # 폰트 크기 비율 적용
    font_path = '/data/ephemeral/home/mlc/resources/NanumGothic.ttf'
    scaled_font_size = max(10, int(font_size * ratio))
    font = ImageFont.truetype(font_path, scaled_font_size)

    # 텍스트 크기 계산
    left, top, right, bottom = font.getbbox(text)
    text_width = right - left
    text_height = bottom - top

    # 텍스트 위치 조정 (이미지 경계 넘지 않도록)
    _, _, x1, y1 = box
    img_width, img_height = pil_img.size
    text_x = min(x1, img_width - text_width)
    text_y = min(y1 + 2, img_height - text_height - 2)  # y1 기준 약간 아래쪽으로 위치

    # 텍스트 그리기 (PIL)
    draw.text((text_x, text_y), text, font=font, fill=color)

    # PIL(RGB) -> OpenCV(BGR) 변환
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def paint_ocr(image, items, thickness=2, ratio=1.0, font_size=10):
  img = np.array(image)
    
  if ratio != 1.0:
    height, width = img.shape[:2]
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    img = cv2.resize(img, (new_width, new_height))
      
  for i in range(len(items)):
    box = items[i].box * ratio
    color = get_random_color()
    img = paint_bound_box(img, box, color, thickness)
    img = paint_text(img, items[i].text, box, color, ratio, font_size=font_size)
      
  return img


def paint_bound_box(image, box, color, thickness=10):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    bbox = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
    return cv2.polylines(image, [pts], isClosed=True, color=color, thickness=thickness)

def resize_image_with_ratio(image_path, ratio=1.0):
    image = Image.open(image_path)
    width, height = image.size

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def resize_image_with_max(image_path, max_len=1024):
    image = Image.open(image_path)
    width, height = image.size

    # 비율 유지하여 최대 길이가 max_len가 되도록 계산
    if width > height:
        scale_factor = max_len / float(width)
        new_width = max_len
        new_height = int(height * scale_factor)
    else:
        scale_factor = max_len / float(height)
        new_height = max_len
        new_width = int(width * scale_factor)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image





'''
--------------------------------------------------
LEGACY
--------------------------------------------------
'''
def get_words_and_boxes_legacy(image, meta, ds_code):
    if ds_code == 'd1':
        items = meta['annotations']
        text_key = 'annotation.text'
        get_box = lambda it: from_d1_box(it['annotation.bbox'])
    elif ds_code == 'd2':
        items = meta['bbox']
        text_key = 'data'
        get_box = lambda it: from_d2_box(it['x'], it['y'])
    elif ds_code == 'd3':
        items = meta['annotations'][0]['polygons']
        text_key = 'text'
        get_box = lambda it: from_d3_box(it['points'])
    else:
        raise ValueError(f"Unknown ds_code: {ds_code}")

    words, boxes = [], []
    for it in items:
        norm_box = to_norm_box(image, get_box(it))
        boxes.append(norm_box)
        words.append(it[text_key])

    return words, boxes

def write_doc_classes_in_json(ds_dir_path):
    # all json paths
    meta_paths = grep_files(ds_dir_path, exts=["json"])

    classes = {get_doc_class_with_ds_code(load_json(p))[0] for p in meta_paths}
    classes = sorted(list(classes))

    # document classes set
    dc = { 'document_classes' : classes }

    # write json file
    with open("doc_classes.json", "w", encoding="utf-8") as f:
        json.dump(dc, f, ensure_ascii=False, indent=4)

def get_doc_class_with_ds_code(meta):
    if 'Identifier' in meta:
        doc_name = meta['images'][0]['document_name']
        dc = f"D3_{doc_name}"
        return dc, 'd3' 
    
    elif 'Annotation' in meta: # 4000
        industries = {0: '은행', 1: '보험', 2: '증권', 3: '물류' }   # 기타는 모두 물류
        ind_idx = meta['Images'].get('form_industry', 3)
        dc = f'D2_{industries[ind_idx]}_{meta['Images']['form_type']}'
        return dc, 'd2'

    dc = f"D1_{meta['images'][0]['image.category']}"
    return dc, 'd1'
    
def from_d1_box(box):
    # [225 1011 326 59] 
    x,   y = box[0:2]
    dx, dy = box[2:]
    return [x, y, x+dx, y+dy]  

def from_d2_box(x, y):
    # x: [1163, 1163, 1348, 1348]
    # y: [450,   508,  450,  508]
    # to x0, y0, x1, y1
    return [x[0], y[0], x[2], y[1]]
    
def from_d3_box(points):
    # [[299, 221], [457, 219], [456, 264], [298, 267]]
    # to x0, y0, x1, y1
    points_np = np.array(points)
    min_x = int(np.min(points_np[:, 0]))
    max_x = int(np.max(points_np[:, 0]))
    min_y = int(np.min(points_np[:, 1]))
    max_y = int(np.max(points_np[:, 1]))
    
    return [min_x, min_y, max_x, max_y]

'''
d1
'''
def to_d1_example(image, meta):
    words = []
    boxes = []
    boxes_origin = []
    annots = meta['annotations']
    
    for an in annots:
        box = an['annotation.bbox']
        bbox = convert_box_to_layoutlm_box(image, box)
        boxes.append(bbox)
        boxes_origin.append(box)

        word = an['annotation.text']
        words.append(word)
        
    category = get_category(meta)

    return image, category, words, boxes, boxes_origin


'''
D2
'''
def to_d2_example(image, meta):
    words = []
    boxes = []
    boxes_origin = []
    
    bbox_list = meta['bbox']
    for item in bbox_list:
        x, y = item['x'], item['y']
        box = d2_xy_to_box(x, y)
        bbox = convert_box_to_layoutlm_box(image, box)
        boxes.append(bbox)
        boxes_origin.append(box)

        word = item['data']
        words.append(word)
        
    category = get_category(meta)

    return image, category, words, boxes, boxes_origin

def d2_xy_to_box(x, y):
    # "x": [ 1163, 1163, 1348, 1348 ],
    # "y": [ 450, 508, 450, 508 ]
    # to x, y, dx, dy
    return [x[0], y[0], x[2]-x[0], y[1]-y[0]]
    


'''
D3
'''
def to_d3_example(image, meta):
    words = []
    boxes = []
    boxes_origin = []
    
    bbox_list = meta['annotations'][0]['polygons']
    for item in bbox_list:
        points = item['points']
        box = d3_xy_to_box(points)
        bbox = convert_box_to_layoutlm_box(image, box)
        boxes.append(bbox)
        boxes_origin.append(box)

        word = item['text']
        words.append(word)
        
    category = get_category(meta)
    return image, category, words, boxes, boxes_origin
    
def d3_xy_to_box(points):
    # [ [ 299.14072340030594, 221.9974704771915 ],
    # [ 457.6176921214762, 219.77556826813105 ],
    # [ 456.9654051974274, 264.98114981339046 ],
    # [ 298.3947878266497, 267.24326068790936 ] ]
    # to x, y, dx, dy
    points_np = np.array(points)
    min_x = int(np.min(points_np[:, 0]))
    max_x = int(np.max(points_np[:, 0]))
    min_y = int(np.min(points_np[:, 1]))
    max_y = int(np.max(points_np[:, 1]))
    
    x = min_x
    y = min_y
    dx = max_x - min_x
    dy = max_y - min_y
    
    return [x, y, dx, dy]


def easy_sample(sid='1'):
    image = Image.open(f"./sample/d{sid}.jpg").convert("RGB")
    image = ImageOps.exif_transpose(image)
    meta = load_json(f"./sample/d{sid}.json")

    dn = sid[0]
    if dn == '1':
        example = to_d1_example(image, meta)
    elif dn == '2':
        example = to_d2_example(image, meta)
    elif dn == '3':
        example = to_d3_example(image, meta)
    
    image, category, words, boxes, boxes_o = example
    return image, category, words, boxes, boxes_o, meta

def paint_bound_dbox(image, box, color):
  # box format: x, y, dx, dy 
  x0 = int(box[0])
  y0 = int(box[1])
  x1 = int(box[2]) + x0
  y1 = int(box[3]) + y0
  paint_bound_box(image, [x0, y0, x1, y1], color)

def rotate_image_cv2(img, direction='cw'):
    if direction == 'cw':
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'ccw':
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def rotate_image_pil(img, direction='cw'):
    if direction == 'cw':
        # PIL의 rotate는 반시계 방향이므로, 시계 방향은 -90도
        return img.rotate(-90, expand=True)
    elif direction == 'ccw':
        return img.rotate(90, expand=True)
    return img

def prepare_example_legacy(image_path, processor):
    # load image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)  # correct orientation

    # load metas
    json_path = Path(image_path).with_suffix(".json")
    meta = load_json(json_path)

    # document class
    document_class, ds_code = get_doc_class_with_ds_code(meta)

    # words and boxes
    words, boxes = get_words_and_boxes(image, meta, ds_code)
    
    encoding = processor(
        images=image,
        text=words,
        boxes=boxes,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    encoding['doc_class_str'] = document_class 
    
    return encoding
