import os
import cv2
import csv
import json
import shutil
import logging
import random
import numpy as np
import pandas as pd
import itertools

from tqdm.notebook import tqdm
from pathlib import Path
from PIL import Image, ImageOps, ImageFont, ImageDraw

default_trans = {
    "account_number":"계좌번호",
    "application_for_payment_of_pregnancy_medical_expenses": "임신/출산",
    "car_dashboard": "계기판",
    "confirmation_of_admission_and_discharge": "입퇴원 확인서",
    "diagnosis": "진단서",
    "driver_lisence": "면허증",
    "medical_bill_receipts": "의료비 영수증",
    "medical_outpatient_certificate": "진료/통원 확인서",
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
    
def calculate_match_rate(dfs, target_col):
    targets = pd.concat([df[target_col].reset_index(drop=True) for df in dfs], axis=1)
    match = targets.nunique(axis=1) == 1
    match_rate = match.mean()
    return match_rate

def partial_match_rate(dfs, target_col):
    targets = pd.concat([df[target_col].reset_index(drop=True) for df in dfs], axis=1)
    n = len(dfs)
    total_pairs = n * (n - 1) // 2

    def match_rate(row):
        matches = sum(1 for a, b in itertools.combinations(row, 2) if a == b)
        return matches / total_pairs if total_pairs > 0 else 1.0

    rates = targets.apply(match_rate, axis=1)
    return rates, rates.mean()

def vote_df(dfs, target_col):
    df_ans = new_df = dfs[0].copy()
    targets = pd.concat([df[target_col].reset_index(drop=True) for df in dfs], axis=1)
    voted = targets.mode(axis=1)[0]
    df_ans[target_col] = voted.astype(int)
    return df_ans







    