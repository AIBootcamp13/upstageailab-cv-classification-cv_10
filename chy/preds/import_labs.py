from __future__ import annotations

import json
import copy
import shutil
import random
import re
import math
import warnings

from pathlib import Path
from datetime import datetime

import requests
import sys, os
import importlib
from functools import reduce
from functools import wraps
from itertools import pairwise, chain
from collections import defaultdict

from contextlib import nullcontext
from contextlib import contextmanager

from enum import Enum, StrEnum
from types import SimpleNamespace
from dataclasses import dataclass, asdict, field
from typing import Dict, Protocol, Callable, Iterable, List

# vision
import cv2
from PIL import Image

# jupyter
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

# visualization
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.express as px
import seaborn as sns
from ipywidgets import interact, Output
from pandas import DataFrame

# visualization: MAP
import folium
from folium.plugins import HeatMap
from pandas._typing import AggFuncType
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# utils
# from tqdm import tqdm
from tqdm.notebook import tqdm


# scientific
import pickle
import pandas as pd
import numpy as np

# from haversine import haversine_vector, Unit

# algorithm
import difflib
from rapidfuzz import fuzz

# data
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.model_selection import train_test_split

# Model
import ray

# from autogluon.common import space
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# from autogluon.tabular import TabularPredictor
# from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

# metric & measure & transform
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import eli5
from eli5.sklearn import PermutationImportance

"""
 PRE SETUP
"""


def setup_font():
    font_name = r"/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    fe = fm.FontEntry(fname=font_name, name="NanumBarunGothic")

    fm.fontManager.ttflist.insert(0, fe)  # Matplotlib에 폰트 추가
    plt.rcParams.update({"font.size": 10, "font.family": "NanumBarunGothic"})
    plt.rc("font", family="NanumBarunGothic")


warnings.filterwarnings("ignore")
setup_font()


"""
DOCSY
"""
def select_ext_only(paths, ext='.jpg'):
    return [p for p in paths if os.path.splitext(p)[1] == ext]
    
def sanity_check_sample(path):
    p = Path(path)
    img_path = p.with_suffix('.jpg')
    jsn_path = p.with_suffix('.json')
    
    if not is_valid_image(img_path):
        return False
    if not load_json(jsn_path):
        return False
    return True 
    
def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except:
        return None

      
def is_valid_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 이미지의 무결성 검사
        return True
    except Exception:
        return False
        
def has_any_not(paths, ext='.jpg'):
    return any(not p.lower().endswith(ext) for p in paths)

def get_category(meta):
    if 'Annotation' in meta: # 4000
        industries = {0: '은행', 1: '보험', 2: '증권', 3: '물류' }   # 기타 -> 물류
        i_code = meta['Images'].get('form_industry', 3)
        industry = industries[i_code]
        category = meta['Images']['form_type']
        return f'D2_{industry}_{category}'
        
    elif 'Identifier' in meta:
        return f"D3_{meta['images'][0]['document_name']}"
        
    return f"D1_{meta['images'][0]['image.category']}"

def count_class(json_paths):
    cs = set()
    for i, p in tqdm(enumerate(json_paths)):
        try:
            j = load_json(p)
        except:
            print(f"error path: {json_paths[i]}")
            continue
            
        c = get_category(j)
        cs.add(c)
        
    return cs

def copy_with_json(files, archive_path):
    for p in tqdm(files):
        dir_path, filename = os.path.split(p)
        _, dir_name = os.path.split(dir_path)
        name, ext = os.path.splitext(filename)
        img_path = os.path.join(dir_path, f"{name}.jpg")
        lab_path = os.path.join(dir_path, f"{name}.json")

        arc_dir = os.path.join(archive_path, dir_name)
        img_arc_path = os.path.join(arc_dir, f"{name}.jpg")
        lab_arc_path = os.path.join(arc_dir, f"{name}.json")
        # print(img_path, img_arc_path)    
        # print(lab_path, lab_arc_path)    

        os.makedirs(arc_dir, exist_ok=True)
        shutil.copy(img_path, img_arc_path)
        shutil.copy(lab_path, lab_arc_path)

def copy_ds(image_paths, arc_dir_path, chunk_size=1000):
    print(f"데이터셋 이사... 청크 단위: {chunk_size}")
    path_chunks = list_chunk(image_paths, chunk_size)
    
    for chunk_idx in tqdm(range(len(path_chunks))):
        curr_dir = os.path.join(arc_dir_path, f"{chunk_idx:05d}")
        os.makedirs(curr_dir, exist_ok=True)

        for p in path_chunks[chunk_idx]:
            try:
                name = os.path.splitext(os.path.basename(p))[0]
                path = Path(p)
                path_img = os.path.join(curr_dir, f"{name}.jpg")
                path_jsn = os.path.join(curr_dir, f"{name}.json")
                shutil.copy(path.with_suffix('.jpg'), path_img)
                shutil.copy(path.with_suffix('.json'), path_jsn)
            except Exception as e:
                print(f'error in {p}')

"""
 CHOCOS
"""
def load_json(path):
  with open(path) as f:
    return json.load(f)


def sum_total_file_size(paths):
    total = 0
    for p in tqdm(paths):
        total += os.path.getsize(p)
    return total


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


def collect_file_paths(*paths):
    items = {}
    for path in paths:
        files = grep_files(path)
        for p in tqdm(files):
            name = os.path.basename(p)
            if name not in items:
                items[name] = p
    return list(items.values())


def print_code(functor):
    import inspect

    print(inspect.getsource(functor))


def is_numeric(x):
    return isinstance(x, (int, float, np.integer, np.floating))


def try_attr_name(name: str, prefix=None):
    name = name.lower().strip().replace(" ", "_")
    if name[0].isdigit():
        name = "_" + name
    name = re.sub(r"[^\w\s]", "_", name)
    return name if not prefix else prefix + name


def get_day_of_year(year, month, day):
    dt = datetime(year, month, day)
    return dt.timetuple().tm_yday


def haversine(lat_a, lon_a, lat_b, lon_b):
    # 위도, 경도 라디안으로
    lat_a, lon_a, lat_b, lon_b = map(np.radians, [lat_a, lon_a, lat_b, lon_b])
    d_lat = lat_b - lat_a
    d_lon = lon_b - lon_a

    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(d_lon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # 지구 반지름 (km)
    return r * c


def grep_files(dir_path: str, exts=None) -> List[str]:
    if exts is None:
        exts = []

    greps = []
    for root, dirs, files in tqdm(os.walk(dir_path)):
        for file in files:
            greps.append(os.path.join(root, file))

    if not exts:
        return greps

    exts = [e.lower() for e in exts]
    filtered = [f for f in greps if extension(f) in exts]

    return filtered


def to_norm_file_path(path):
    dir_path, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    return os.path.join(dir_path, f"{name}_norm{ext}")


def random_choose(items):
    if not items:
        return None
    return items[np.random.choice(len(items))]


def random_sample_items(items, percentage=0.01):
    target_items = items[:] 
    random.shuffle(target_items)
    return [it for it in target_items if random.random() < percentage]


def extension(path):
    return os.path.splitext(path)[-1][1:].lower()


def show_img(path, by="matplot", figsize=(10, 10)):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    img = mpimg.imread(path)

    if img is None:
        print(f"wrong path: {path}")

    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def filename(path):
    return os.path.splitext(os.path.basename(path))[0]


def save_image_as_jpg(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")
    result = cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not result:
        raise IOError(f"이미지 저장 실패: {output_path}")
    return output_path


def copy_ext_proc(path, idx, out_dir):
    ext = extension(path)
    name = filename(path)

    if ext == "png":
        copy_path = os.path.join(out_dir, f"{name}.jpg")
        save_image_as_jpg(path, copy_path)

    if ext == "jpeg" or ext == "jpg":
        copy_path = os.path.join(out_dir, f"{name}.jpg")
        shutil.copy(path, copy_path)

    if ext == "json":
        shutil.copy(path, out_dir)

def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def merge_dataset(paths, merge_root_dir, chunk_size=1000, start=0, last=-1):
    print("파일 색인 중 (이미지, 레이블)")
    groups = defaultdict(list)
    for p in tqdm(paths):
        name = filename(p)
        groups[name].append(p)

    # shutil.rmtree(merge_root_dir)
    # os.makedirs(merge_root_dir, exist_ok=True)

    names = list(groups.keys())
    start_chunk_idx = start * chunk_size
    last_chunk_idx = len(names) if last == -1 else last * chunk_size

    print(f"데이터셋 통합... 청크 단위: {chunk_size}")
    for idx in tqdm(range(start_chunk_idx, last_chunk_idx, chunk_size)):
        chunk_dir = os.path.join(merge_root_dir, f"{idx//chunk_size+1:05d}")
        os.makedirs(chunk_dir, exist_ok=True)

        name_chunk = names[idx : idx + chunk_size]
        for name in name_chunk:
            files = groups[name]
            for p in files:
                try:
                    copy_ext_proc(p, idx, chunk_dir)
                except Exception as e:
                    print(e)
                    break


"""
 HOW TO
"""


def how_df_csv():
    how = """
    # dataframe csv append
    df.to_csv('파일명.csv', mode='a', header=False, index=False)
    
    # 컬럼 앞에 추가하기
    df.insert(0, 'column_name_to_insert', pd.NA)  # 맨 앞에 컬럼 추가, 값은 결측치
    """
    print(how)


def how_reimport():
    how = """
    importlib.reload(package)
    from import_labs import *
    """
    print(how)


def how_easy_notion(precision=3):
    how = """
    for numpy
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=3, suppress=True
        formatter={'float_kind':'{:0.2f}'.format})
    with np.printoptions(...):
    for pandas
    pd.options.display.float_format
    pd.set_option('display.float_format', '{:.1f}'.format)
    """
    print(how)


def how_show_all():
    how = """
    pd.get_option('display.max_columns')
    pd.set_option('display.max_columns', None)    # 모든 열 표시
    pd.set_option('display.width', None)          # 자동 폭 조정
    pd.set_option('display.max_colwidth', None)   # 셀 내용 생략 없이 표시
    """
    print(how)


def how_pd_encoding():
    how = """
    data = pd.read_csv('some-scv-file.csv', encoding='latin1')
    data: DataFrame
    encoding UnicodeDecodeError
    """
    print(how)


def how_sort():
    how = """
    data: DataFrame
    df.sort_values(by='a', ascending=False)
    df.sort_values(by=['b', 'c'], ascending=[False, True])
    
    data: Series
    series.sort_values(ascending=True, na_position='last')
    """
    print(how)


def how_shuffle():
    how = """
    data: DataFrame
    shuffled = df.sample(frac=1).reset_index(drop=True)
    shuffled = df.iloc[np.random.permutation(df.index)].reset_index(drop=True)
    """
    print(how)


def how_select_cols():
    how = """
    대괄호 리스트\t   df[["col1", "col2"]]\t      컬럼명 선택
    loc\t   df.loc[:, ["col1", "col2"]]\t   행, 열 레이블로 선택
    loc\t   df.loc[:, [T,F,T,...]]\t        행, 열 레이블로 선택
    iloc\t  df.iloc[:, [0, 2, 4]]\t         행, 열 위치(정수 인덱스)로 선택
    
    """
    print(how)


def how_to_filter_series():
    how = """
    pd.Series 0인 항목 제거
    filtered = s[s != 0]
    filtered = s.loc[s != 0]
    zero_indices = s[s == 0].index
    """
    print(how)


def how_to_find_index():
    how = """
    셔플 전 439번 행이 현재 train 프레임에서 몇 번째 행 순번인가?
    ds.train.index.get_loc(439)
    """
    print(how)


def how_eda_phase1():
    how = """
    # 데이터 로드
    ds = DS.from_csv('Global-YouTube-Statistics.csv')
    ds = DS.from_nd_array(dataset['data'], dataset['feature_names'])
    ds = DS.from_csv_train_test(train_path, trial_path, encoding="UTF-8")
    
    # 시드 고정
    ds.easy_notion(3)
    ds.set_random_seed(515)
    
    # 시작 데이터 설정
    ds.merge_train_and_trial()
    ds.make_keyword(Cursor.ORIGIN)
    
    # TARGET 데이터 로드 (별도로 있는 경우)
    ds.setup_target(dataset['target'])   # pd.Series
    
    # 데이터 살펴보기
    ds.print_shape()
    ds.print_columns()
    ds.print_info()
    ds.print_describe()
    ds.print_head(3)
    ds.print_feature(feature_idx=3, use_random=True)
    ds.gen_print_feature()
    ds.print_count_cat_and_con()  # 범주형 vs 연속형 개수
    
    # 범주형 살펴보기
    ds.print_unique_values(ds.key_channel_type)
    ds[ds.key_channel_type].unique()
    ds[ds.key_channel_type].value_counts()
    ds.print_value_counts(ds.key_channel_type)
    
    # 랜덤 샘플 (행) 살펴보기
    ds.print_row(use_random=True)
    
    # 대략적 기술 통계량 살펴 보기 
    ds.gen_describe()       # 전체 넘기면서 보기
    print_describe()        # 전체 살펴보기
    print_describe(col_key) # 특정 컬럼 살펴보기
    """
    print(how)


def how_notna():
    how = """
    # (NOTNA) col1, col2, col3 컬럼에 대해 샘플에 na이 하나도 없는 행만 선택
    mask_notna = df[['col1', 'col2', 'col3']].notna().all(axis=1)
    df_notna = data[mask_notna]
    """
    print(how)


def how_eda_phase1_na():
    how = """
    # 결측치 살펴보기
    ds.print_isnull_count(axis=0, use_sort=True, drop_zero=True)
    ds.num_of_nan_row(ds.key_k_복도유형)  # 해당 컬럼의 결측 수

    # 결측치 행, 열 기준으로 살펴보기
    ds.rows_with_nans(1, only_nan_cols=True)
    ds.cols_with_nans_less(3)
    
    # 반응형 위젯으로 결측율과 카테고리 (고유값)를 확인
    ds.interact_value_counts(only_cat=False)
    
    # 특정 컬럼의 결측치가 있는 행을 (모든 컬럼 정보로) 가져오기
    - target 정보는 train에는 missing 될 수 없으므로 아래 결과는 empty
    ds.rows_with_nan_in_col(ds.key_target, mark=Mark.ON_TRAIN)  
    ds.rows_with_nan_in_col(ds.key_target, mark=Mark.ON_ASIS)   # cursor 그대로 

    NOTE: 결측치 수가 유사한 피쳐들이 있다면 동시에 발생하여
          특정 행들에 공통적으로 결측된 것인지 체크
    ds.count_rows_all_missing_in_cols(cols_co_nan)

    # 의미론적 결측치 체크 (명시적 결측치가 아니지만) 
    g = ds.gen_unique_category_values()      # 범주형 체크
    g = ds.gen_describe(only_num_cols=True)  # 수치형 체크
    next(g)     # 넘기면서 보기

    # 기타 복합 조건들은 논리 검사, 집계함수 활용
    (ds[ds.key_subscribers] < 0).any()
    """
    print(how)


def how_eda_phase1_extended():
    how = """
    # 데이터 살펴보기 인터랙티브 EXTENDED
    ds.interact_unique(only_cat=False)  # 고유값 탐험
    
    # 컬럼 데이터 타입 변환
    ds.convert_type(key, as_type=str, inplace=True)
    
    # 값에 대응되는 모든 행 출력 
    - only_key: 해당 카테고리만 출력
    ds.match_rows(ds.key_k_전화번호, 28150900.0, only_key=True) 
    
    """
    print(how)


def how_eda_phase2():
    how = """
    # 데이터 이상치 확인 (박스 플롯)
    ds.draw_box_plot(ds.key_column_name)
    g = ds.gen_draw_box_plot()
    
    NOTE: 상자가 작고 선 위에 많은 수염 점들이 있는 경우 (right long-tail 분포)
    NOTE: 분석 목표에 따라 IQR 이상치라도 제거하지 않을 수 있음 (유튜브 수익 등)
    
    # 데이터 분포 확인 (히스토그램 w/ 로그 척도 변환)
    ds.draw_histogram(key)
    g = ds.gen_draw_histogram()
    ds.draw_histogram(key, log_scale=(False,True))   # 렌더링 버그 있음
    
    NOTE: 위 두 그래프로 쓰나미 찾기 
    - LONG-TAIL, RIGHT-SKEWED 분포의 경우 y축 log-scale 척도변환 
    - 작은 데이터 윤곽 드러나 시각화 도움
    
    # 범주별 데이터 분석 (BAR 차트 범주형 -> 연속형 매핑)
    - 막대 그래프로 범주형 컬럼의 각 항목별 연속형 변수에 매핑 (평균 계산)
    - 예시) 유튜브 주제별 월 소득
    ds.category_cols() 
    ds.draw_bar_chart(ds.key_category, ds.key_lowest_monthly_earnings)
    
    # 상관 계수와 상관 행렬 (correlation matrix)
    corr, mask = make_corr_matrix_with_mask(data):
    draw_corr_matrix(figsize=(7, 5)):
    
    NOTE: 상관성이 높은 경우 해당 변수가 단지 타 변수의 다른 표현인 공선성 가능성 체크
    - 다중 공선성 (multi-co-linearity): 한 변수가 평균값 등 다른 여러 변수의 조합인 경우 
    """
    print(how)


def how_eda_phase3():
    how = """
    # 데이터 전처리
    ds.activate_cleaning()  # cursor.CLEANING 모드 진입
    
    # 임시 작업을 위한 커서 컨텍스트 매니져 제공
    with ds.with_cursor(Cursor.ORIGIN):
        print(ds.cursor_mode)   # Cursor.ORIGIN
        ds.print_describe()
    print(ds.cursor_mode)   # 원래 커서 모드로 복귀
    
    # 결측치 처리 
    ## 한 행에 결측치 n개 이상인 행들 제거
    row_ids = ds.row_ids_with_nans_less(8)
    ds.clean_by_keep_rows(row_ids)
    
    ## nan 대체 처리
    ds.imputation_fillna("대체 가능한 컬럼", 0)
    
    ## 나머지 nan 행들 드롭
    ds.dropna(axis=0) 
    ds.print_nan_count()  # 결측치 없음 체크
    
    """
    print(how)


def how_eda_phase4():
    how = """
    # 데이터 분포 변환 (함수변환, 스케일링, 구간화)
    
    # 주의
    - 변환이 모델링에 있어서 역효과 일 수 있으므로 롤백 대비할 것 (클리닝 모드)
    - 변환 후 상황에 따라 역변환(원본 척도)이 필요한 경우가 있음 
    - 학습 데이터 뿐만 아니라 테스트, 타겟 데이터 등도 반영 필요한 경우 있음 ><
    
    # 클리닝 모드 진입 (Cursor.CLEANING)
    ds.activate_cleaning()
    mean, std = ds.mean_and_std_per_feature()   # 피쳐별 평균, 표준편차 계산
    
    # 커서 데이터 전체에 대해 표준화 수행 적용
    ds.apply_standard_on_cursor()   
    
    # TARGET에도 표준화 수행 적용
    with ds.with_cursor(Cursor.TARGET):
        ds.apply_standard_on_cursor()
        ds.print_describe()
    
    # 표준화 수행 결과 비교 (easy_notion 후 볼 것)
    ds.compare_describe_with(Cursor.ORIGIN)  # compare ORIGIN VS CLEANING
    
    # 파생 피쳐 추가하기 (로그 스케일링)  
    ds.add_derived_log_transform(key)           # 추가키: key_log_{origin}
    ds.compare_describe(key_a, key_log_a)
    ds.draw_bar_chart_compares(key_a, key_log_a)
    
    # 데이터 스케일 변환 (정규화, 표준화)
    - 서로 다른 변수들 간의 척도를 통일시켜 각 변수간 영향력 공평하게 (분포는 유지)
    
    # 파생 피쳐 추가하기 (표준화 standardization)
    # 공식: scaled_x = (x - mean(x)) / std(x)
    add_derived_standard_transform(key):
    
    # 파생 피쳐 추가하기 (정규화 normalization, min-max scaling)
    # 공식: scaled_x = (x - min(x)) / max(x)
    add_derived_minmax_norm_transform(key):
    """
    print(how)


def how_eda_feature_engineer():
    how = """
    # 실험 전 TEMP mode 활용 (기존 모드 데이터 복제)
    ds.activate_temp()
    ... temp cursor 에서 실험과 그래프
    ds.deactivate_temp()  # 기존 모드 복귀
    
    # 컬럼 타입 통일성 체크
    ds.ux_dtype_unique()    
    ds.show_all_mix_dtype()            # 한번에 보기
    df[K.key_번지].map(type).unique()  # series에서 타입 모두 보기
    
    # 컬럼 내 데이터 타입 통일 (혼합 타입 정렬 실패 해결)
    ds.convert_type(ds.key_아파트명, as_type=str)
    
    # 파생 변수 사전 체크 (데이터 눈도장)
    ds.sample(K.key_계약년월, n=3)
    
    # 파생 변수 사전 체크 (조건 체크) 
    ds.query_rows_in_col(ds.key_시군구, lambda x: x.split()[0] != "서울특별시")   # 모두 서울시인가?
    ds.map(ds.key_시군구, lambda x: x.split()[1])  # mapper 테스트
    
    ds.query_rows_in_col(K.key_계약년월, lambda x: len(str(x)) != 6)  # 모두 6자리 인가?
    ds.map(ds.key_계약년월, lambda x: str(x)[:4])  # 변환 함수 체크    
    
    # 파생 변수 생성
    ds.add_feature_by_map(feat_name="구", col_key=ds.key_시군구, mapper=lambda x: x.split()[1])
    ds.add_feature_by_map(feat_name="동", col_key=ds.key_시군구, mapper=lambda x: x.split()[2])
    
    # 변수 빈도 사전 체크 (max_rows=none for all rows view)
    ds.ux_value_counts(DataType.ALL, max_rows=None)
    
    # 변수 변환 (n개 미만 아파트명 변경)
    ds.query_rows_in_col(K.key_아파트명, lambda x: "롯데" in x)
    
    # 변수 삭제 (데이터 상 일정 빈도 이하 삭제)
    ds.drop_by_freq_cond(K.key_아파트명, num_freq=3)
    
    """
    print(how)


def how_model_split():
    how = """
    # l1 노말라이즈 ([8, 4, 2] => [8/14, 4/14, 2/14])
    nl = normalize(lst, norm="l1")  # np.2D array 형태 요구
    
    # 학습 데이터 분할
    ds.split_dataset(ratios=[0.7, 0.3], with_target=True)
    ds.print_dataset_split()    # shape and ratio
    """
    print(how)


def how_model_linear_regress():
    how = """
    # (다중) 선형회귀모델 
    regressor = ds.fit_linear_regressor()   # TRAIN
    ds.draw_regressor_result(regressor)
    ds.report_regressor_parameter(regressor)
    
    # 선형회귀 해석적 해법
    theta = (X^TX)^-1@X^Ty analytic solution이 알려져 있음
    
    # 학습한 수치와 해석적 계산 수치에 아주 미세한 차이 있음
    regressor = ds.analytic_linear_solver()   # regressor protocol 충실 함수 리턴
    ds.report_regressor_parameter(regressor)  # protocol 만족하면 사용 가능
    
    # 주의
    - 데이터셋의 셔플 랜덤성에 따라 intercept, coefficients 완전 다를 수 있음
    
    # 결정 계수와 상관 계수 증명
    - 표준화를 한 경우 단순 선형 회귀의 계수는 상관 계수와 같음을 증명
    - statproofbook.github.io/P/slr-rsq.html
    """
    print(how)


def how_df_apply_map():
    how = """
    map	for Series	
    - Series의 각 요소에 함수, 딕셔너리, 매퍼 등을 적용
    - 단순 매핑에 적합
    - DataFrame에는 직접 사용 불가
    apply for Series, DataFrame	
    - Series: 각 요소에 함수 적용 (map과 유사)
    - DataFrame: 행(row) 또는 열(column) 단위로 함수 적용, 복잡한 연산 가능
    - axis 옵션으로 행/열 지정
    applymap for DataFrame	
    - DataFrame의 모든 개별 원소(셀)에 함수 적용
    - 각 셀에 동일한 연산을 일괄 적용할 때 사용
    """
    print(how)


def how_df():
    how = """
    # 다양한 조건의 마스크 OR
    mask_doro_na = df[K.key_도로명].isna() | (df[K.key_도로명].str.strip() == "") | (df[K.key_도로명].str.isdecimal())
    mask_coord_na = df[K.key_좌표x].isna()
    
    # str.strip()에 수치 체크하려면 (주의)
    df[K.key_도로명].str.strip().str.isdecimal()
    
    # 두 마스크 공통 조건
    num_both = (mask_doro_na & mask_coord_na).sum()
    
    # 조건 XOR (^)
    ds[K.key_좌표x].isna() ^ ds[K.key_좌표y].isna()  # 좌표는 x, y 모두 존재 혹은 둘 다 없음
    
    # 두 조건 모두 만족하는 케이스 (좌표는 없으면서 주소도 없는 경우)
    mask_empty_coord & mask_wrong_addr
    
    # 값 쓰기 (복사본 주의)
    df.loc[idx, K.key_좌표x] = cx   # OK (원본에 써짐)
    df.loc[idx][K.key_좌표x] = cx   # FAILED Series 복제본에 써버림
    
    # 특정 컬럼의 행값을 키로 딕셔너리 생성
    m = df.set_index('key_col')[['col1', 'col2', 'col3']].apply(list, axis=1).to_dict()
    
    # 딕셔너리 생성
    m = df[['col1', 'col2', 'col3']].apply(list, axis=1).to_dict()
    
    # 특정 행부터 컬럼 대응값을 한칸씩 밀기 (반복 실행 주의)
    for idx in range(9042, len(df)-1):
        df.iloc[idx] = df.iloc[idx].shift(1)
    """
    print(how)


def how_df_select_rows_unique_counts():
    how = """
    # >>> 훈련셋, 테스트셋 분포 비교 (아파트 소량 건수) 
    
    # 데이터 준비
    df = ds.merged
    df = df[df[K.key_is_trial] == 1]
    sr = df[K.key_아파트명]

    # 아파트명 기준 발생 건수 카운트
    counts = sr.value_counts()
    
    # 발생 건수 2회 미만 인덱스 (인덱스는 아파트명의 배열)
    cut = counts.quantile(0.1)   # 카운트 하위 10% 기준으로 함
    idx = counts[counts <= cut].index
    
    # 각 행이 인덱스에 해당 하는지 sr 전체 행에 T/F 플래그
    mask = sr.isin(idx)
    
    # 조건 맞는 행 선택
    df_res = df[mask]
    
    # 상대적 비율 조사
    100 * df_res.shape[0] / sr.shape[0]
    
    # 테스트셋 소량건 아파트가 훈련셋에 포함된 비율 조사
    sr_train  = df[df[K.key_is_trial] == 0][K.key_아파트명]
    idx_train = srt.value_counts().index 
    
    # 교집합을 통한 체크 
    idx.intersection(idx_train).shape[0] / idx.shape[0]
    """
    print(how)


def how_df_value_between():
    how = """
    # 좌표 바운드 조건으로 필터링 하는 법 (between 활용)
    min_lat, max_lat = 37.413294, 37.715133
    min_lon, max_lon = 126.734086, 127.269311
    df[df["좌표X"].between(min_lon, max_lon) & df["좌표Y"].between(min_lat, max_lat)] 
    """
    print(how)


"""
DECORATOR & GENERATOR 
"""


@contextmanager
def with_display_opts(width=120, max_cols=20, max_rows=100, max_col_w=50):
    width_val = pd.options.display.width
    max_col_w_val = pd.options.display.max_colwidth
    max_rows_val = pd.options.display.max_rows
    max_cols_val = pd.options.display.max_columns
    try:
        pd.options.display.max_rows = max_rows
        pd.options.display.max_columns = max_cols
        pd.options.display.max_colwidth = max_col_w
        pd.options.display.width = width
        yield
    finally:
        pd.options.display.width = width_val
        pd.options.display.max_colwidth = max_col_w_val
        pd.options.display.max_rows = max_rows_val
        pd.options.display.max_columns = max_cols_val


@contextmanager
def with_display_max_rows():
    max_rows_val = pd.options.display.max_rows
    try:
        pd.options.display.max_rows = None
        yield
    finally:
        pd.options.display.max_rows = max_rows_val


def track_rows_changed():
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            prev_num_rows = len(self.cursor)
            result = func(self, *args, **kwargs)
            next_num_rows = len(self.cursor)
            print(f"num rows changed: {prev_num_rows-next_num_rows}")
            print(f"[prev] num rows: {prev_num_rows}")
            print(f"[next] num rows: {next_num_rows}")
            return result

        return wrapper

    return decorator


"""
 Dataset Class
"""


class Cursor(Enum):
    TRAIN = "train"
    VALID = "valid"
    TRIAL = "trial"
    ORIGIN = "origin"
    MERGED = "merged"
    CLEANING = "cleaning"
    TARGET = "target"
    TEMP = "temp"
    ALL = "all"

    def __str__(self):
        return self.value


class Mark(StrEnum):
    ON_ASIS = "on_asis"
    ON_TRAIN = "on_train"
    ON_TRIAL = "on_trial"
    IS_TRIAL = "is_trial"

    def __str__(self):
        return self.value


class DataType(Enum):
    ALL = "all"
    CAT = "category"
    CON = "continuous"

    def __str__(self):
        return self.value


class Symbols(StrEnum):
    REPLACE_STR = "Unknown"
    Preds = "preds"
    Target = "target"
    RMSE = "rmse"


class Tools:
    def __init__(self, ds: DS):
        self.ds = ds
        self.key_to_transform = {}

    @staticmethod
    def draw_price_geo_heatmap(data: pd.DataFrame, width="800", height="600"):
        min_lat, max_lat = 37.413294, 37.715133
        min_lon, max_lon = 126.734086, 127.269311
        seoul_center = [37.511619900016576, 127.02149035854055]

        f = folium.Figure(width=width, height=height)
        m = folium.Map(
            location=seoul_center,
            zoom_start=11,
            min_lat=min_lat,
            min_lon=min_lon,
            max_lat=max_lat,
            max_lon=max_lon,
            tiles="CartoDB Positron",
        ).add_to(f)

        # 데이터 필터링 (위도, 경도, 가격 모두 존재하는 항목)
        mask_notna = data[["좌표X", "좌표Y", "target"]].notna().all(axis=1)
        df = data[mask_notna]

        # 가격 표준화 (0 ~ 1 범위로 알아서 정규화 한다고 함 (필요 없을 듯)
        # scaled_target = DS.scale_standard(df["target"])
        # df["target"] = pd.DataFrame(data=scaled_target, columns=["target"])

        # 동일 좌표
        df["count"] = df.groupby(["lat", "long"]).cumcount() + 1

        # HeatMap 데이터 준비 (위도, 경도, 값 순서)
        heat_data = [[row["좌표Y"], row["좌표X"], row["target"]] for idx, row in df.iterrows()]

        HeatMap(
            heat_data,
            min_opacity=0.4,
            radius=10,
            blur=10,
            max_zoom=18,
            gradient={"0.2": "blue", "0.4": "lime", "0.6": "yellow", "0.8": "red"},
        ).add_to(m)

        return m

    @staticmethod
    def get_address_info(keyword):
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        headers = {"Authorization": f"KakaoAK 3a3673d5fecda4d3fc5363e6f99f4bfd"}
        response = requests.get(url, headers=headers, params={"query": keyword})
        result = response.json()

        if "errorType" in result:
            print(f"errorType: {result['errorType']}")
            return None
        try:
            return result["documents"][0]
        except IndexError:
            return None

    @staticmethod
    def get_address_keys(data: pd.DataFrame):
        df = data
        not_empty_doro = df["도로명"].notna() & (df["도로명"] != "")
        unique_road_name = df.loc[not_empty_doro, "도로명"].unique()
        unique_addr_name = (df.loc[~not_empty_doro, "시군구"] + " " + df.loc[~not_empty_doro, "번지"]).unique()
        return np.hstack((unique_addr_name, unique_road_name))

    def make_address_map(self, keys):
        address_map = {}
        for keyword in keys:
            info = self.get_address_info(keyword)
            if not info:
                print(f"좌표 찾기 실패: {keyword}")
            else:
                address_map[keyword] = Tools.flatten_dict(info)
        return pd.DataFrame(address_map).transpose()

    @staticmethod
    def save_dict_to_csv(data: Dict, filename: str):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

    @staticmethod
    def flatten_dict(data: Dict, parent_key="", sep="."):
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(Tools.flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                for idx, item in enumerate(v):
                    idx_key = f"{new_key}[{idx}]"
                    if isinstance(item, dict):
                        items.extend(Tools.flatten_dict(item, idx_key, sep=sep).items())
                    else:
                        items.append((idx_key, item))
            else:
                items.append((new_key, v))
        return dict(items)

    def fill_xy(self, df: pd.DataFrame):
        no_addr = {}
        for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="주소 채우기"):
            # key_addr = row[K.key_도로명].strip()
            # key_lega = row[K.key_시군구]+" "+row[K.key_번지]
            #
            # if key_addr=="" or key_addr.isdecimal():
            #     key_addr = key_lega
            #
            # if key_addr in addr_mapper:
            #     key_addr = addr_mapper[key_addr]
            # if key_lega in addr_mapper:
            #     key_lega = addr_mapper[key_lega]
            #
            # if key_addr in no_addr and key_lega in no_addr:
            #     continue
            # if key_addr not in dict_addr_doro and key_addr not in dict_addr_lega and key_lega not in dict_addr_doro and key_lega not in dict_addr_lega:
            #     no_addr[key_addr] = 1
            #     no_addr[key_lega] = 1
            #     continue
            #
            # if key_lega in dict_addr_lega:
            #     coord = dict_addr_lega[key_lega]
            # elif key_lega in dict_addr_doro:
            #     coord = dict_addr_doro[key_lega]
            # elif key_addr in dict_addr_lega:
            #     coord = dict_addr_lega[key_addr]
            # elif key_addr in dict_addr_doro:
            #     coord = dict_addr_doro[key_addr]
            #
            # cx, cy = coord
            # ds.train.loc[idx, K.key_좌표x] = cx
            # ds.train.loc[idx, K.key_좌표y] = cy
            pass

    @staticmethod
    def calc_day_of_year(row: pd.Series):
        return get_day_of_year(int(row["계약연도"]), int(row["계약월"]), int(row["계약일"]))

    @staticmethod
    def calc_days_since_first_day(row: pd.Series) -> int:
        # 편의상 데이터셋의 최초 년도의 첫 날짜를 기준
        dt_ctr = datetime(int(row["계약연도"]), int(row["계약월"]), int(row["계약일"]))
        dt_ref = datetime(2007, 1, 1)
        return (dt_ctr - dt_ref).days

    @staticmethod
    def drop_geo_outlier(data: pd.DataFrame, drop_na=False) -> pd.DataFrame:
        df = data
        min_lat, max_lat = 37.413294, 37.715133
        min_lon, max_lon = 126.734086, 127.269311

        return df[df["좌표X"].between(min_lon, max_lon) & df["좌표Y"].between(min_lat, max_lat)]

    def apply_support_data(self):
        ds = self.ds
        K = ds.keys
        """
        - 좌표와 도로명 보조 데이터 적용
        - 좌표 이상치: 서울시 외부 제거
        """
        use_cols = {K.key_도로명, K.key_좌표x, K.key_좌표y}
        df_train_xy = pd.read_csv("./train_xy.csv", usecols=use_cols)
        df_trial_xy = pd.read_csv("./trial_xy.csv", usecols=use_cols)

        with ds.with_cursor(Cursor.TRAIN):
            ds.overwrite_by_df(df_train_xy)
        with ds.with_cursor(Cursor.TRIAL):
            ds.overwrite_by_df(df_trial_xy)

        # 강남구 127.0495556 위경도  37.514575
        ds.train.fillna({K.key_좌표x: 37.514575, K.key_좌표y: 127.0495556}, inplace=True)
        ds.trial.fillna({K.key_좌표x: 37.514575, K.key_좌표y: 127.0495556}, inplace=True)
        # 좌표 결측치까지 날림
        ds.storage[Cursor.TRAIN] = Tools.drop_geo_outlier(ds.train)
        # ds.storage[Cursor.TRIAL] = Tools.drop_geo_outlier(ds.trial)
        print(f"train 좌표 결측# :{ds.train[K.key_좌표x].isna().sum()}")
        print(f"trial 좌표 결측# :{ds.trial[K.key_좌표x].isna().sum()}")

    def feature_engineer(self, for_ts=True):
        ds = self.ds
        K = ds.keys
        """
        - mixed dtype 정리
        - 계약년월: 년/월 분리
        - 시군구: 구/동 분리
        - 아파트명: 결측치(2136 nan) -> '보통' 아파트
        - 가격 (target) 스케일 변환
        """
        # 컬럼 mixed dtype 정리 (쓰이는 것 만)
        ds.convert_as_str_type(K.key_아파트명)
        ds.convert_as_str_type(K.key_도로명)

        # 시군구 -> 구, 동 변환
        ds.add_feature_by_map(feat_name="구", col_key=K.key_시군구, mapper=lambda x: x.split()[1])
        ds.add_feature_by_map(feat_name="동", col_key=K.key_시군구, mapper=lambda x: x.split()[2])

        # 계약년월 -> 계약연도, 계약월
        ds.add_feature_by_map(feat_name="계약연도", col_key=K.key_계약년월, mapper=lambda x: str(x)[:4])
        ds.add_feature_by_map(feat_name="계약월", col_key=K.key_계약년월, mapper=lambda x: str(x)[4:])

        # day-of-year (계약월 + 계약일)
        sr_ctr_doy = ds.apply(Tools.calc_days_since_first_day, axis=1)
        ds.add_feature(data=sr_ctr_doy, col_name="계약거리일")

        # cursor (merged)
        df = self.ds.cursor

        # 평수 - 면적 round 처리 후 변환
        sr_areas = df[K.key_전용면적___]
        df_pys = sr_areas.map(lambda x: np.round(2 * x / 3.30579) / 2)
        ds.add_feature(data=df_pys, col_name="평수")

        # 가격 스케일 변환
        key_scaled_target = f"scaled_{K.key_target}"
        df_scaled = self.transform_log1p_std_scale(df[K.key_target], key_scaled_target)
        ds.add_feature(data=df_scaled, col_name=key_scaled_target)

        # 좌표 x, y (apply support 과정에서 결측 날라감)
        sr_x, sr_y = df[K.key_좌표x], df[K.key_좌표y]
        key_scaled_x = f"scaled_{K.key_좌표x}"
        key_scaled_y = f"scaled_{K.key_좌표y}"
        df_scaled_x = self.transform_std_scale(sr_x, K.key_좌표x)
        df_scaled_y = self.transform_std_scale(sr_y, K.key_좌표y)
        ds.add_feature(data=df_scaled_x.squeeze(), col_name=key_scaled_x)
        ds.add_feature(data=df_scaled_y.squeeze(), col_name=key_scaled_y)

        # key refresh
        K = ds.keys

        # scaled x, y + 평수 K-means
        key_x, key_y, key_z = K.key_scaled_좌표x, K.key_scaled_좌표y, K.key_평수
        X = df[[key_x, key_y, key_z]]
        kms = KMeans(n_clusters=200, random_state=self.ds.random_seed)
        xyp_labels = kms.fit_predict(X)
        key_xypy = "cluster_xypy"
        ds.add_feature(data=xyp_labels, col_name=key_xypy)

        # 최종 타입 지정 (범주형, 수치형)
        ds.storage[Cursor.MERGED][K.key_층] = ds[K.key_층].astype(float)
        ds.storage[Cursor.MERGED][K.key_건축년도] = ds[K.key_건축년도].astype(float)
        ds.storage[Cursor.MERGED][K.key_cluster_xypy] = ds[K.key_cluster_xypy].astype(str)
        if not for_ts:
            ds.storage[Cursor.MERGED][K.key_계약거리일] = ds[K.key_계약거리일].astype(float)
            ds.storage[Cursor.MERGED][K.key_계약연도] = ds[K.key_계약연도].astype(float)
            ds.storage[Cursor.MERGED][K.key_계약월] = ds[K.key_계약월].astype(float)

    def feature_engineer_heavy(self):
        # 위경도 기반 apt to sub/bus 거리 계산
        df_bus = pd.read_csv("../../data/bus_feature.csv")
        df_sub = pd.read_csv("../../data/subway_feature.csv")

        df = self.ds.merged
        K = self.ds.keys

        df_apt = df[[K.key_좌표y, K.key_좌표x]].drop_duplicates()
        cs_apt = list(zip(df_apt[K.key_좌표y], df_apt[K.key_좌표x]))
        cs_bus = list(zip(df_bus["Y좌표"], df_bus["X좌표"]))
        cs_sub = list(zip(df_sub["위도"], df_sub["경도"]))

        # distance matrix apt to sub (apt-to-bus 49GB 메모리 필요)
        # dm_apt_sub = haversine_vector(cs_apt, cs_sub, Unit.METERS, comb=True).reshape(len(df_apt), len(df_sub))
        # dm_apt_bus = haversine_vector(cs_apt, cs_bus, Unit.METERS, comb=True).reshape(len(df_apt), len(df_bus))

        # 플립된 포트란 결과를 내주기 때문에 T(transpose) 후 reshape 적용해야 함
        # dm_aptu_sub = haversine_vector(scs_apt, scs_sub, Unit.KILOMETERS, comb=True).T.reshape(
        #     len(scs_apt), len(scs_sub), order="F"
        # )

    def post_drop_and_resplit(self, for_ts=True):
        ds = self.ds
        K = ds.keys
        """
        # 제거
        - 시군구, 계약년월, 계약일, 전용면적, 
        # 제거 대기
        ['좌표X', '좌표Y', 'target']
        # 리스트
        ['층', '건축년도', '도로명', '좌표X', '좌표Y', 'target', '구', '동', '계약연도', '계약월', '계약거리일', '평수']
        """
        if not for_ts:
            ds.drop_keys([K.key_시군구, K.key_계약년월, K.key_계약일, K.key_전용면적___])
        else:
            ds.drop_keys([K.key_시군구, K.key_전용면적___])

        ds.resplit_to_train_and_trial()

    def make_scaler_for_key(self, key: str):
        if key in self.key_to_transform:
            raise Exception(f"[{key}] already has transform")
        scaler = self.key_to_transform[key] = StandardScaler()
        return scaler

    def del_scaler_for_key(self, key: str):
        if key not in self.key_to_transform:
            raise Exception(f"[{key}] does not have transform")
        del self.key_to_transform[key]

    def select_price_quantile_in_year(self, df, year, q_pair=(0.8, 0.9)):
        K = self.ds.keys

        years = df[K.key_계약연도].unique()
        years.sort()
        year_to_df = {int(y): df[df[K.key_계약연도] == str(y)] for y in years}

        df_year = year_to_df[int(year)]
        df_year_price = df_year[K.key_target]

        qx, qy = q_pair
        lower_q = int(df_year_price.quantile(qx))
        upper_q = int(df_year_price.quantile(qy))
        mask = (df_year_price >= lower_q) & (df_year_price <= upper_q)

        return df_year[mask]

    def transform_std_scale(self, data: pd.DataFrame | pd.Series, key) -> np.ndarray:
        scaler = self.make_scaler_for_key(key)
        df = data if isinstance(data, pd.DataFrame) else data.to_frame()
        nd_arr = scaler.fit_transform(df)
        print(f"scaler meta [{key}]: mean:{scaler.mean_} scale:{scaler.scale_}")
        return nd_arr

    def transform_log1p_std_scale(self, sr: pd.Series, key: str) -> pd.Series:
        df = np.log1p(sr.to_frame())
        nd_arr = self.transform_std_scale(df, key)
        return pd.Series(nd_arr.flatten(), index=sr.index)

    def inverse_log1p_std_scale(self, sr: pd.Series, key: str) -> pd.DataFrame | None:
        if key not in self.key_to_transform:
            print(f"[{key}] 키로 저장된 변환 없음")
            return None
        scaler: StandardScaler = self.key_to_transform[key]
        return np.expm1(scaler.inverse_transform(sr.to_frame()))

    def ux_bar_chart(self, figsize=(13, 10), sort_x=True, no_x_tick=False):
        def draw_chart(output: Output, col_x, col_y):
            with output:
                clear_output(wait=True)
                df = self.ds.cursor
                if self.ds.cursor_mode == Cursor.MERGED:
                    df = df[df[Mark.IS_TRIAL] == 0]
                if sort_x:
                    df = df.sort_values(by=[col_x, col_y], ascending=True)
                self.ds.draw_bar_chart_from(df, col_x, col_y, figsize, no_x_tick)

        self.ds.interact_two_column(draw_chart)

    def kmeans_cluster(self, df: pd.DataFrame, key_x, key_y, n_clusters=10):
        X = df
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.ds.random_seed)
        kmeans_labels = kmeans.fit_predict(X)

        plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, s=30, cmap="viridis")
        plt.scatter(
            kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", marker="x", s=200, label="Centroids"
        )
        plt.xlabel(f"{key_x}")
        plt.ylabel(f"{key_y}")
        plt.title("K-means Clustering Result")
        plt.legend()
        plt.show()


class DS:
    __RANDOM_SEED__ = 0
    __PRELOAD_COLS__ = None

    def __init__(self):
        self.random_seed = 0
        self.storage = defaultdict(pd.DataFrame)
        self._cursor_mode = Cursor.ORIGIN
        self._prev_cursor_mode = None
        self.split_source_cursor = None
        self.keys = SimpleNamespace()
        self.tools = Tools(self)

    @classmethod
    @contextmanager
    def with_selected_cols(cls, use_cols: list[str]):  # TODO: rename with_cols_selector()
        cls.__PRELOAD_COLS__ = use_cols
        try:
            yield
        finally:
            cls.__PRELOAD_COLS__ = None

    @classmethod
    def from_csv(cls, path: str, encoding="latin1") -> DS:
        ds = cls()
        ds.cursor = pd.read_csv(path, encoding=encoding)
        return ds

    @classmethod
    def from_csv_train_test(cls, path_train, path_trial, sample=False, encoding="latin1") -> DS:
        # None: 모두 읽음, functor: 없는 컬럼은 무시함, list: 없는 컬럼 있으면 에러남
        cols_selector = lambda c: c in cls.__PRELOAD_COLS__
        cols = cols_selector if cls.__PRELOAD_COLS__ else None

        if sample:
            train = DS.read_csv_sampling(path_train, num_sample=10000, cols=cols)
        else:
            train = pd.read_csv(path_train, encoding=encoding, usecols=cols)
        trial = pd.read_csv(path_trial, encoding=encoding, usecols=cols)

        ds = cls()
        ds.storage[Cursor.TRAIN] = train
        ds.storage[Cursor.TRIAL] = trial
        ds.cursor_mode = Cursor.TRAIN

        return ds

    @classmethod
    def read_csv_sampling(cls, csv_path: str, num_sample=10000, cols=None):
        with open(csv_path, "r") as f:
            num_lines = sum(1 for _ in f) - 1
        skip = sorted(random.sample(range(1, num_lines + 1), num_lines - num_sample))
        return pd.read_csv(csv_path, skiprows=skip, usecols=cols)

    @classmethod
    def from_nd_array(cls, data, feature_names: list) -> DS:
        ds = cls()
        ds.cursor = pd.DataFrame(data, columns=feature_names)
        return ds

    @staticmethod
    def set_random_seed(key: int):
        np.random.seed(key)
        DS.__RANDOM_SEED__ = key

    def pre_setup(self):
        self.easy_notion(2)
        self.apply_strip_str()
        self.make_keyword()
        # TODO
        # - verbose mode decorator (verbose -> print every Cursor[cursor_mode]))
        # - apply decorator in DS.every method
        note = """ TODO:
        - 데이터 augmentation
        - chart 인자 (with_figsize 처리)
        """
        print(note)

    """
    Cursor
    """

    def of(self, cursor: Cursor) -> pd.DataFrame:
        return self.storage[cursor]

    @property
    def cursor(self) -> pd.DataFrame:
        return self.storage[self.cursor_mode]

    @cursor.setter
    def cursor(self, data: pd.DataFrame):
        # this setter doesn't handle self.cursor[col] = data_ (NO)
        # self.cursor = data (OK)
        self.storage[self.cursor_mode] = data

    @property
    def cursor_mode(self):
        return self._cursor_mode

    @cursor_mode.setter
    def cursor_mode(self, mode: Cursor):
        self._cursor_mode = mode
        print(f">>> changed cursor mode: [{mode}]")

    @contextmanager
    def with_cursor(self, cursor: Cursor):
        back_cursor_mode = self.cursor_mode
        self.cursor_mode = cursor
        try:
            yield
        finally:
            self.cursor_mode = back_cursor_mode

    @property
    def train(self) -> pd.DataFrame:
        return self.of(Cursor.TRAIN)

    @property
    def valid(self) -> pd.DataFrame:
        return self.of(Cursor.VALID)

    @property
    def trial(self) -> pd.DataFrame:
        return self.of(Cursor.TRIAL)

    @property
    def target(self) -> pd.DataFrame:
        return self.of(Cursor.TARGET)

    @property
    def origin(self) -> pd.DataFrame:
        return self.of(Cursor.ORIGIN)

    @property
    def merged(self) -> pd.DataFrame:
        return self.of(Cursor.MERGED)

    """
    ACCESS
    """

    def __getitem__(self, item) -> pd.DataFrame | pd.Series:
        key_values = self.keys.__dict__.values()
        if item in key_values:
            return self.cursor[item]  # cursor["column-key"]
        elif type(item) is Cursor:
            return self.of(item)  # storage[Origin|Train|...]
        elif type(item) is list and len(item) > 0 and item[0] in key_values:
            # cursor[['key-a', 'key-b]] -> DataFrame not Series
            return self.cursor[item]
        return None

    def __setitem__(self, key, value):
        if type(key) is Cursor:
            self.storage[key] = value

    def sample_in_column(self, col_key: str, n=10):
        return self.cursor[col_key].sample(n)

    def sample(self, n=10, columns=None):
        # filtered = df.dropna(subset=cols)
        # sampled = filtered.sample(n=원하는개수, random_state=시드)
        if columns is None:
            return self.cursor.sample(n)
        return self.cursor[columns].sample(n)

    def apply(self, func: AggFuncType, axis=0):
        return self.cursor.apply(func=func, axis=axis)

    """
    TEMP MODE
    """

    @contextmanager
    def with_temp(self):
        self.activate_temp()  # UX 차트 함수는 컨텍스트 밖에서 수행되어 적용 어려움
        try:
            yield
        finally:
            self.deactivate_temp()

    def activate_temp(self):
        if Cursor.TEMP == self.cursor_mode:
            print(f"cursor mode: [{self.cursor_mode}] already")
            return
        self._prev_cursor_mode = self.cursor_mode
        print(f"stash cursor mode: [{self._prev_cursor_mode}]")
        df_temp = self.cursor.copy(deep=True)
        self.cursor_mode = Cursor.TEMP
        self.cursor = df_temp

    def deactivate_temp(self):
        print(f">>> rollback to... <<<")
        self.cursor_mode = self._prev_cursor_mode
        print(f"<<< rollback complete >>>")

    def on_temp_mode(self, data: pd.DataFrame):
        self.cursor_mode = Cursor.TEMP
        self.cursor = data

    def drop_by_cond(self, col_key: str, cond):
        # s = self.cursor[col_key]
        # ns = s.shape[0]
        # c = s.apply(cond)
        # nc = c.shape[0]
        # self.cursor[col_key] = s[~c].reset_index(drop=True)
        c = self.get_filtered_column(col_key, cond)
        self.cursor[col_key] = c

        nc = c.shape[0]
        ns = self.cursor.shape[0]
        print(f"num rows: {ns} -> {self.cursor.shape[0]}")
        print(f"num drop: {nc}  drop rate: ({100 * nc / ns:.02f}%)")
        # df = df.query('Age >= 25')
        # df = df.query('col.str.len() < 5', engine='python')

    def get_filtered_column(self, col_key: str, cond):
        s = self.cursor[col_key]
        c = s.apply(cond)
        return s[~c].reset_index(drop=True)

    def drop_by_freq_cond(self, col_key: str, num_freq):
        s = self.cursor[col_key]
        ns = s.shape[0]
        counts = s.value_counts()
        low_freq = counts[counts < num_freq].index
        rows_to_drop = s.isin(low_freq)
        nc = rows_to_drop.sum()
        self.cursor = self.cursor[~rows_to_drop].reset_index(drop=True)

        print(f"num rows: {ns} -> {self.cursor.shape[0]}")
        print(f"num drop: {nc}  drop rate: ({100 * nc / ns:.02f}%)")

    """
    DATA
    """

    def apply_strip_str(self):
        self.cursor = self.cursor.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    def merge_train_and_trial(self):
        """
        합치는 이유
        - 분할된 데이터셋에 대해 일괄 처리를 위해
        - 다시 분리가 필요하므로 식별 요소를 추가
        """
        self.train[Mark.IS_TRIAL] = 0
        self.trial[Mark.IS_TRIAL] = 1
        self.add_key(Mark.IS_TRIAL)
        self.cursor_mode = Cursor.MERGED
        self.cursor = pd.concat([self.train, self.trial])
        self.cursor.reset_index(drop=True, inplace=True)

    def resplit_to_train_and_trial(self):
        if self.cursor_mode != Cursor.MERGED:
            print(f"cursor not on MERGED but [{self.cursor_mode}]")
            return

        df = self.merged
        self[Cursor.TRAIN] = df[df[Mark.IS_TRIAL] == 0]
        self[Cursor.TRIAL] = df[df[Mark.IS_TRIAL] == 1]
        # self.drop_keys({Mark.IS_TRIAL})

    def set_target(self, data: pd.Series):
        with self.with_cursor(Cursor.TARGET):
            self.cursor = pd.DataFrame(data, columns=["target"])

    def overwrite_by_df(self, data: pd.DataFrame):
        # data에 존재하는 컬럼들에 대해 업데이트
        cols = data.columns
        self.cursor[cols] = data

    """
    KEYS
    """

    def backup_keys(self) -> SimpleNamespace:
        return copy.deepcopy(self.keys)

    def make_keyword(self, target=None, show=False):
        if target is None:
            target = self.cursor_mode
        data = self.storage[target]
        for col_name in data.columns.values:
            self.add_key(col_name)
        if show:
            display(self.keys)

    def add_key(self, col_name):
        attr_name = try_attr_name(col_name, prefix="key_")
        if not attr_name or hasattr(self.keys, attr_name):
            print(f">>> key conflicted: [{attr_name}]")
        else:
            setattr(self, attr_name, col_name)
            setattr(self.keys, attr_name, col_name)

    def drop_keys(self, drop_keys: Iterable, target=Cursor.ALL, verbose=False):
        keys_to_del = []
        for key in drop_keys:
            res = self.del_key_by_col(key, verbose)
            if res:
                keys_to_del.append(key)

        if target == Cursor.ALL:
            for store in self.storage.values():
                store.drop(columns=keys_to_del, inplace=True, errors="ignore")
        else:
            self.storage[target].drop(columns=keys_to_del, inplace=True, errors="ignore")

    def del_key_by_col(self, col_name, verbose=False) -> bool:
        key_items = self.keys.__dict__.items()
        key_attrs = [k for k, v in key_items if v == col_name]
        if len(key_attrs) == 0:
            return False

        key_attr = key_attrs[0]
        if hasattr(self, key_attr):
            delattr(self, key_attr)
        if hasattr(self.keys, key_attr):
            delattr(self.keys, key_attr)
        if verbose:
            print(f"key attr deleted: {col_name} : [{col_name}]")
        return True

    """
    COLUMNS & ROWS
    """

    def rename_columns(self, mapper: Dict):
        self.cursor.rename(columns=mapper)

    @property
    def columns(self):
        return self.cursor.columns.tolist()

    def columns_by(self, data_type: DataType):
        if data_type == data_type.CAT:
            return self.category_cols()
        elif data_type == data_type.CON:
            return self.continuous_cols()
        return self.columns

    def continuous_cols(self):
        return [c for c in self.cursor.columns if self[c].dtype.name != "object"]

    def category_cols(self):
        return [c for c in self.cursor.columns if self[c].dtype.name == "object"]

    def columns_with_cardinality(self, data_type: DataType) -> tuple[list, list]:
        cols = self.columns_by(data_type)
        cards = [len(self[c].unique()) for c in cols]
        return cols, cards

    def rows_by_cols(self, col_names: list[str], row_slice: slice) -> pd.DataFrame:
        return self.cursor.loc[row_slice, col_names]

    def rows_by_value(self, col_name, value, only_key=False):
        df = self.cursor
        df = df[df[col_name] == value]
        return df if not only_key else df[col_name]

    """
    DataFrame & Series
    """

    def sort_series(self, data: pd.Series, inplace=False):
        try:
            return data.sort_values(inplace=inplace)
        except:
            data = data.astype(str)  # float, str 혼합 케이스
            sorted_data = data.sort_values(inplace=inplace)
        return sorted_data

    def get_unique(self, col_key: str, sort=False):
        unique = pd.Series(self.cursor[col_key].unique())
        return unique if sort else self.sort_series(unique, inplace=False)

    def get_value_counts(self, col_key: str, sort=False):
        counts = self.cursor[col_key].value_counts()
        return counts if sort else self.sort_series(counts, inplace=False)

    def get_dtype_in_col(self, col_key: str) -> list:
        return self.cursor[col_key].apply(lambda x: type(x).__name__).unique().tolist()

    """
    TYPE
    """

    def convert_as_str_type(self, key: str, use_fillna=False, replace=Symbols.REPLACE_STR):
        types = self.get_dtype_in_col(key)
        if len(types) == 1 and "str" in types:
            return
        elif len(types) == 2 and "float" in types:
            self.cursor[key] = self.cursor[key].astype(str)
            return
        else:
            print(f"not implement yet {types}")
            return

    def change_col_type(self, key: str, as_type):
        self.cursor[key] = self.cursor[key].astype(as_type)

    """
    VISUALIZE
    """

    def easy_notion(self, n=2, scientific=False):
        # DS 클래스에 구현하여 상태 유지하고 필요할 때 마다 변경하는 식으로
        float_format = f"{{:.{n}e}}" if scientific else f"{{:.{n}f}}"
        pd.set_option("display.precision", n)
        pd.set_option("display.float_format", float_format.format)
        np.set_printoptions(precision=n, suppress=scientific)

    """
    EDA: Phase1
    """

    def print_shape(self):
        print(f"[{self.cursor_mode}] shape: {self.cursor.shape}")

    def print_head(self, num=10):
        display(f"{self.cursor_mode} dataset")
        display(self.cursor.head(num))

    def print_columns(self, in_horizontal=True, as_list=False):
        display(f"[{self.cursor_mode}] #cols = {len(self.cursor.columns)}")
        cols = df_columns = pd.DataFrame(self.cursor.columns, columns=["columns"])
        if as_list:
            cols = df_columns.values.flatten()
        elif in_horizontal:
            cols = df_columns.transpose()
        display(cols)

    def print_describe(self, key=None, show_max=False):
        # with with_display_opts(show_max):
        data = self.cursor if key is None else self.cursor[key]
        display(data.describe())

    def gen_describe(self, only_num_cols=False):
        def gen():
            cols = self.continuous_cols() if only_num_cols else self.cursor.columns
            for c in cols:
                print(f"컬럼명: {c}")
                yield self[c].describe()

        return gen()

    def print_info(self):
        display(self.cursor.info())

    def print_count_cat_and_con(self):
        """show num categorical & continuous"""
        num_cate = self.cursor.select_dtypes(include=["object", "category"]).shape[1]
        num_cont = self.cursor.select_dtypes(include=["number"]).shape[1]
        return {"categorical": num_cate, "continuous": num_cont}

    def print_unique_values(self, key, sort=True):
        unique = self.get_unique(key, sort)
        print(f"범주: {unique}")
        print(f"범주수: {unique.shape[0]}")

    def gen_unique_category_values(self):
        def gen():
            for c in self.cursor.columns:
                if self[c].dtype.name != "object":
                    continue
                unique = self[c].unique()
                print(f"컬럼명: {c}")
                print(f"범주수: {len(unique)}")
                yield unique

        return gen()

    def print_value_counts(self, key, sort=True):
        display(self.get_value_counts(col_key=key, sort=sort))

    def print_feature(self, feature_idx=0, use_random=False):
        if use_random:
            feature_idx = np.random.choice(len(self.cursor.columns))
        feature_name = self.cursor.columns[feature_idx]
        print(f"feature index: {feature_idx}")
        display(self.cursor[feature_name])

    def gen_print_feature(self):
        def gen():
            for fi, c in enumerate(self.cursor.columns):
                self.print_feature(fi)
                yield fi

        return gen()

    def print_row(self, row_idx=0, use_random=False):
        if use_random:
            row_idx = np.random.choice(len(self.cursor))
        print(f"row index: {row_idx}")
        display(self.cursor.iloc[row_idx])

    """
    EDA: Phase1.NAN
    """

    def rows_with_nans(self, num_nan: int, only_nan_cols=False):
        ids = self.cursor.isna().sum(axis=1) >= num_nan
        if only_nan_cols:
            cols = self.cursor.isna().sum() > 0
            return self.cursor.loc[ids, cols]
        return self.cursor[ids]

    def rows_with_nan_in_col(self, col_key, mark=Mark.ON_ASIS) -> pd.DataFrame:
        df = self.cursor
        if mark == Mark.ON_TRIAL:
            df = df[df[Mark.IS_TRIAL] == 1]
        elif mark == Mark.ON_TRAIN:
            df = df[df[Mark.IS_TRIAL] == 0]
        mask = df[col_key].isna()
        return df[mask]

    def num_of_nan_row(self, col_key):
        return self.cursor[col_key].isna().sum()

    def cols_with_nans_less(self, num_nan: int):
        cols = self.cursor.isna().sum(axis=0) < num_nan
        return self.cursor.loc[:, cols]

    def count_rows_all_missing_in_cols(self, col_keys):
        return self.cursor[col_keys].isnull().all(axis=1).sum()

    def row_ids_with_nans_less(self, num_nan: int):
        return self.cursor.isna().sum(axis=1) < num_nan

    def search_pseudo_nans(self):
        """비정상 범위, 자연수 타입에서 음수, 무의미 문자"""
        pass

    def print_isnull(self):
        display(self.cursor.isna())  # isnull === isna

    def print_isnull_count(self, axis=0, use_sort=True, drop_zero=False):
        """axis 0:column-wise 1:row-wise"""
        sr = self.cursor.isna().sum(axis)
        if use_sort:
            sr = sr.sort_values(ascending=False)
        if drop_zero:
            sr = sr[sr != 0]
        display(sr)

    """
    EDA: Phase2
    """

    @staticmethod
    def make_corr_matrix_with_mask(data: pd.DataFrame):
        corr = data.corr(numeric_only=True)
        mask = np.ones_like(corr, dtype=bool)
        mask = np.triu(mask)
        return corr, mask

    def draw_box_plot(self, key):
        """
        이상치 파악 용이 IQR 활용 그래프
        박스 높이: IQR
        박스 내부의 수평선: 중앙값
        박스 외부 수평선: 최소(Q1 - 1.5*IQR), 최대(Q3 + 1.5*IQR)
        수염(whisker) - 이상치
        """
        sns.boxplot(data=self.cursor, y=key)
        plt.show()

    def gen_draw_box_plot(self):
        for c in self.continuous_cols():
            display(f"컬럼 {c}")
            self.draw_box_plot(c)
            yield c

    def calc_quantile(self, key):
        col = self.cursor[key]
        q1 = col.quantile(q=0.25)
        q3 = col.quantile(q=0.75)
        iqr = q3 - q1
        return q1, q3, iqr

    def draw_histogram(self, key, log_scale=(False, False), kde=True, horizontal=False, figsize=(10, 10)):
        df = self.cursor.sort_values(key)
        kwargs = {"y": key} if horizontal else {"x": key}
        plt.figure(figsize=figsize)
        sns.histplot(data=df, kde=kde, log_scale=log_scale, **kwargs)
        plt.xticks(rotation=90)
        plt.show()

    def gen_draw_histogram(self):
        for c in self.continuous_cols():
            display(f"컬럼 {c}")
            self.draw_histogram(c)
            yield c

    def draw_bar_chart_from(self, data: pd.DataFrame, key_x, key_y, figsize=(13, 10), no_x_tick=False):
        plt.figure(figsize=figsize)
        bar_chart = sns.barplot(data=data, x=key_x, y=key_y, color="C0", errorbar=None)
        if not no_x_tick:
            loc, labels = plt.xticks()
            bar_chart.set_xticklabels(labels, rotation=90)
        plt.title(f"{key_y} per each {key_x}")
        plt.show()

    def draw_bar_chart(self, key_x, key_y, figsize=(13, 10)):
        self.draw_bar_chart_from(self.cursor, key_x, key_y, figsize=figsize)

    def draw_corr_matrix(self, figsize=(13, 10)):
        corr, mask = DS.make_corr_matrix_with_mask(self.cursor)
        plt.figure(figsize=figsize)
        sns.heatmap(data=corr, annot=True, fmt=".2f", mask=mask, linewidths=0.5, cmap="RdYlBu_r")
        plt.title("Correlation Matrix")
        plt.show()

    def draw_line_chart_from(self, data: pd.DataFrame, key_x: str, key_y: str, figsize=(13, 10)):
        df = data
        df = df[key_x].sort_values()

        plt.figure(figsize=figsize)
        sns.lineplot(data=df, x=key_x, y=key_y)
        plt.title("--- title? ---")
        plt.show()

    """
    UX INTERACTIVE
    """
    UxOneColFuncType = Callable[[Output, str], None]
    UxTwoColFuncType = Callable[[Output, str, str], None]

    def interact_one_column(self, functor: UxOneColFuncType, data_type: DataType):
        def on_column_change(change):
            if change["type"] == "change" and change["name"] == "value":
                item_str: str = change["new"]
                col_name = item_str[: item_str.index(":")]
                functor(output, col_name)

        output = widgets.Output()
        cols, cards = self.columns_with_cardinality(data_type)
        drop_items = [f"{x}: (항목 {y}개)" for x, y in zip(cols, cards)]

        col_dropdown = widgets.Dropdown(
            options=drop_items,
            value=drop_items[0],
            description="컬럼 :",
        )

        col_dropdown.observe(on_column_change, names="value")
        display(col_dropdown, output)

    def ux_box_plot(self, data_type=DataType.ALL):
        def box_plot(output: Output, col_name):
            with output:
                clear_output(wait=True)
                self.draw_box_plot(col_name)

        self.interact_one_column(box_plot, data_type)

    def ux_hist(self, log_scale=(False, False), kde=True, horizontal=False, figsize=(15, 15)):
        def draw_chart(output: Output, col_name):
            with output:
                clear_output(wait=True)
                self.draw_histogram(col_name, log_scale, kde, horizontal, figsize)

        self.interact_one_column(draw_chart, DataType.ALL)

    def ux_unique(self, data_type=DataType.ALL, as_list=True):
        def show_unique_values(output: Output, col_name):
            unique = self.get_unique(col_name, sort=True)
            unique = unique.tolist() if as_list else unique

            with output:
                clear_output(wait=True)
                print(f"'{col_name}' 컬럼의 고유값: ({len(unique)}개)")
                # with with_display_opts(max_rows=None if n < 500 else 100):
                print(unique) if as_list else display(unique)

        self.interact_one_column(show_unique_values, data_type)

    def ux_dtype_unique(self):
        def show_unique_values(output: Output, col_name):
            with output:
                clear_output(wait=True)
                unique = self.get_dtype_in_col(col_name)
                print(f"'{col_name}' 타입 통일: {'YES' if len(unique) == 1 else 'NO'}")
                display(unique)

        self.interact_one_column(show_unique_values, DataType.ALL)

    def ux_value_counts(self, data_type=DataType.ALL, max_rows=120):
        def show_value_counts(output: Output, col_name):
            # counts: pd.Series = self.cursor[col_name].value_counts()
            counts = self.get_value_counts(col_name)
            num_nan = self.num_of_nan_row(col_name)
            n = len(self.cursor)
            num_not_nan = n - num_nan

            # 각 카테고리 비율 표기 (카운트 / (전체 - 결측수))
            counts = counts.apply(lambda x: f"{x}  ({100 * x / num_not_nan:.02f}%)")
            with output:
                clear_output(wait=True)
                print(f"'{col_name}' 컬럼 고유값 ({len(counts)}개):")
                print(f"\t결측값 ({num_nan} / {n})")
                print(f"\t결측율 ({(100 * num_nan) / n:.02f}%)\n")
                with with_display_opts(max_rows=max_rows):
                    display(counts)

        self.interact_one_column(show_value_counts, data_type)

    def interact_two_column(self, functor: UxTwoColFuncType):
        def on_column_change(change):
            if change["type"] != "change":
                return

            dx, dy = drop_x.value, drop_y.value
            if dx == "미선택" or dy == "미선택":
                return

            col_x = dx[: dx.index(":")]
            col_y = dy[: dy.index(":")]
            functor(output, col_x, col_y)

        output = widgets.Output()
        cols, cards = self.columns_with_cardinality(DataType.ALL)
        drops = ["미선택"] + [f"{x}: (항목 {y}개)" for x, y in zip(cols, cards)]

        drop_x = widgets.Dropdown(options=drops, value=drops[0], description="컬럼 :")
        drop_y = widgets.Dropdown(options=drops, value=drops[0], description="컬럼 :")
        drop_x.observe(on_column_change, names="value")
        drop_y.observe(on_column_change, names="value")

        display(drop_x, drop_y, output)

    def ux_bar_chart(self, figsize=(13, 10), sort_x=True, no_x_tick=False):
        # TODO
        #  - 차트 렌더링 시 x축 데이터 항목당 개수 표시
        #  - x축 조건이나 슬라이더로 원하는 카테고리만 선택
        #  - 조건에 해당하는 총 데이터 수 / 미결측 총 데이터 제목에 표기
        def draw_chart(output: Output, col_x, col_y):
            with output:
                clear_output(wait=True)
                df = self.cursor
                if self.cursor_mode == Cursor.MERGED:
                    df = df[df[Mark.IS_TRIAL] == 0]
                if sort_x:
                    df = df.sort_values(by=[col_x, col_y], ascending=True)
                self.draw_bar_chart_from(df, col_x, col_y, figsize, no_x_tick)

        self.interact_two_column(draw_chart)

    """
    FEATURE ENGINEERING 
    """

    def show_all_mix_dtype(self):
        col_to_dtype = []
        for col in self.columns:
            dtypes = self.get_dtype_in_col(col)
            if len(dtypes) > 1:
                col_to_dtype.append((col, dtypes))
        s = pd.Series({k: ", ".join(v) for k, v in col_to_dtype})
        display(s)

    def query_rows_in_col(self, col_key: str, mapper):
        mask = self.map(col_key, mapper)
        return self.cursor[mask]

    def map(self, col_key: str, mapper):
        return self.cursor[col_key].map(mapper)

    def add_feature_by_map(self, feat_name: str, col_key: str, mapper):
        data = self.map(col_key, mapper)
        self.add_feature(feat_name, data)

    def add_feature(self, col_name: str, data: pd.Series | pd.DataFrame):
        if col_name in self.columns:
            print(f"col_name already used: {col_name}")
            return
        sr = data if isinstance(data, pd.Series) else data.squeeze()
        self.cursor[col_name] = sr
        self.add_key(col_name)

    """
    EDA: Phase3: N/A
    """

    def activate_cleaning(self):
        if self.cursor_mode is Cursor.CLEANING:
            return
        self[Cursor.CLEANING] = self.cursor.copy()
        self.cursor_mode = Cursor.CLEANING

    @track_rows_changed()
    def clean_by_keep_rows(self, row_ids):
        self.cursor = self.cursor[row_ids]

    def imputation_fillna(self, key, value):
        self[key].fillna(value, inplace=True)

    @track_rows_changed()
    def dropna(self, axis=0):
        """결측치 포함된 axis=0 행 드롭,  axis=1 열 드롭"""
        self.cursor.dropna(axis=axis, inplace=True)

    def print_nan_count(self):
        sr_nan = self.cursor.isnull().sum()
        print(f"dataset[{self.cursor_mode}] #nan: {sr_nan.sum()}")
        display(sr_nan)

    def draw_na(self, figsize=(13, 2)):
        plt.figure(figsize)
        df = self.cursor
        missing = df.isnull().sum() / df.shape[0]
        missing = missing[missing > 0]
        missing.sort_values(inplace=True)
        missing.plot.bar(color="blue")
        plt.title("변수별 결측치 비율")
        plt.show()

    def mean_and_std_per_feature(self):
        mean_per_ft = self.cursor.mean(axis=0)
        std_per_ft = self.cursor.std(axis=0)
        return mean_per_ft, std_per_ft

    def apply_standard_on_cursor(self):
        mean, std = self.mean_and_std_per_feature()
        self.cursor = (self.cursor - mean) / std

    def compare_describe_with(self, cursor_x: Cursor):
        with self.with_cursor(cursor_x):
            self.print_describe()
        self.print_describe()

    """
    EDA: Phase4
    """

    @staticmethod
    def scale_standard(data: pd.DataFrame | pd.Series) -> tuple[pd.DataFrame, StandardScaler]:
        if isinstance(data, pd.Series):
            data = data.to_frame()
        scaler = StandardScaler()
        return scaler.fit_transform(data), scaler

    @staticmethod
    def scale_minmax_normalize(data: pd.DataFrame):
        """data type must be pd.DataFrame not pd.Series"""
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)

    def compare_describe(self, *key_args):
        display(self.cursor[list(key_args)].describe())

    def draw_histogram_multi(self, *key_args, num_cols=2, figsize=(15, 6)):
        keys = list(key_args)
        num_rows = math.ceil(len(keys) / num_cols)
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=figsize)
        for i, key in enumerate(keys):
            sns.histplot(data=self.cursor, x=key, kde=True, ax=ax[i])
            ax[i].set_title(f"{key}")

        plt.show()

    """
    MODEL: split
    """

    def concat_column_wise(self, cursor_x: Cursor, cursor_y: Cursor):
        x = self.storage[cursor_x].copy()
        y = self.storage[cursor_y].copy()
        return pd.concat([x, y], axis=1)

    @staticmethod
    def _sampling_from_data(data: pd.DataFrame, ratios: list[float], use_random: bool):
        indexes = data.index
        if use_random:
            indexes = np.random.permutation(indexes)

        ratios = np.array(ratios) / np.sum(ratios)
        split_points = (np.cumsum(ratios) * len(data)).astype(int)

        results = []
        sequence = chain([None], split_points)
        for pair in pairwise(sequence):
            ids = indexes[pair[0] : pair[1]]
            results.append(data.loc[ids])
        return results

    def split_dataset(self, ratios=None, with_target=False, use_random=True):
        if ratios is None:
            ratios = [0.6, 0.3, 0.1]

        data = self.cursor
        if with_target:
            data = self.concat_column_wise(self.cursor_mode, Cursor.TARGET)
        self.split_source_cursor = self.cursor_mode  # 어떤 데이터로 분할했는지

        splits = DS._sampling_from_data(data, ratios, use_random)
        self.storage[Cursor.TRAIN] = splits[0]
        self.storage[Cursor.TRIAL] = splits[-1]
        if len(splits) == 3:
            self.storage[Cursor.VALID] = splits[1]

    def print_dataset_split(self):
        n_data = len(self[self.split_source_cursor])
        train = self.of(Cursor.TRAIN)
        valid = self.of(Cursor.VALID)
        trial = self.of(Cursor.TRIAL)
        print(f"SPLIT from: {self.split_source_cursor}")
        print(f"TRAIN: {train.shape}, 비율 {train.shape[0] /n_data:.2f}")
        print(f"VALID: {valid.shape}, 비율 {valid.shape[0] /n_data:.2f}")
        print(f"TRIAL: {trial.shape}, 비율 {trial.shape[0] /n_data:.2f}")

    """
    AutoML: AutoGluon
    """
    #
    # def fit_ts_autogluon(self, target_column: str) -> TimeSeriesPredictor:
    #     import ray
    #
    #     num_cpus = 62  # htop 개수랑 다름 (ray status 확인)
    #     ray.init(dashboard_port=30390, num_cpus=num_cpus)  #  opened port on ai-stage my server
    #
    #     ts_data = []
    #
    #     predictor = TimeSeriesPredictor(
    #         prediction_length=12,
    #         target="price",
    #         known_covariates_names=["interest_rate"],  # 동적 피처
    #         static_features=["size_sqm"],  # 정적 피처
    #     ).fit(ts_data)
    #
    #     memory_limit = 36.0  # 서버 실행 시 변경
    #     time_limit = 2 * 60 * 60  # 버그?: 시간 지정 안하면 3600 제한
    #     train_data = self.train
    #     trial_data = self.trial
    #
    #     return predictor
    #
    # def fit_tabular_autogluon(self, target_column: str) -> TabularPredictor:
    #     num_cpus = 62  # htop 개수랑 다름 (ray status 확인)
    #     ray.init(dashboard_port=30390, num_cpus=num_cpus)  #  opened port on ai-stage my server
    #
    #     memory_limit = 36.0  # 서버 실행 시 변경
    #     time_limit = 2 * 60 * 60  # 버그?: 시간 지정 안하면 3600 제한
    #     train_data = self.train
    #     trial_data = self.trial
    #
    #     predictor = TabularPredictor(label=target_column, eval_metric="rmse").fit(
    #         train_data,
    #         test_data=trial_data,
    #         time_limit=time_limit,
    #         presets="best_quality",
    #         num_gpus=1,
    #         num_cpus=num_cpus,
    #         memory_limit=memory_limit,
    #     )
    #
    #     return predictor

    """
    MODEL: linear regressor
    """

    class LinearRegressorProtocol(Protocol):
        def predict(self, x) -> list: ...

        @property
        def coef_(self) -> int: ...

        @property
        def intercept_(self) -> int: ...

    def check_row_consistency_with_target(self):
        """
        TODO 원본으로 랜덤 샘플 행에서 TRAIN, VALID <-> TARGET 일치 체크
        - 셔플 과정으로 인덱스 불일치
        - 분포 변환, 스케일 등으로 척도 불일치 등
        """
        return self.cursor == 0

    def separate_data_and_target(self, cursor: Cursor, target_key: str):
        df = self.of(cursor)
        target = df[[target_key]]
        features = df.drop(target_key, axis=1)
        return features, target

    def fit_linear_regressor(self) -> LinearRegression:
        target_key = self.target.columns[0]
        train_data, train_target = self.separate_data_and_target(Cursor.TRAIN, target_key)

        regressor = LinearRegression()
        regressor.fit(train_data, train_target)
        return regressor

    def report_regressor_parameter(self, regressor: LinearRegressorProtocol):
        print("intercept: ", regressor.intercept_)
        print("coefficients: ", regressor.coef_)

        target_key = self.target.columns[0]
        train, train_target = self.separate_data_and_target(Cursor.TRAIN, target_key)
        trial, trial_target = self.separate_data_and_target(Cursor.TRIAL, target_key)

        multi_train_pred = regressor.predict(train)
        multi_trial_pred = regressor.predict(trial)

        multi_train_mse = mean_squared_error(multi_train_pred, train_target)
        multi_trial_mse = mean_squared_error(multi_trial_pred, trial_target)
        print(f"TRAIN MSE: {multi_train_mse:.5f}")
        print(f"TRIAL MSE: {multi_trial_mse:.5f}")

    def draw_regressor_result(self, regressor: LinearRegressorProtocol):
        plt.figure(figsize=(4, 4))
        plt.title(f"regressor line for {Cursor.TRIAL}")
        plt.xlabel("target")
        plt.ylabel("prediction")

        trial, trial_target = self.separate_data_and_target(Cursor.TRIAL, self.target.columns[0])
        y_pred = regressor.predict(trial)
        plt.plot(trial_target, y_pred, ".")

        x = np.linspace(-2.5, 2.5, 10)
        y = x
        plt.plot(x, y)

        plt.show()

    def analytic_linear_solver(self) -> LinearRegressorProtocol:
        target_key = self.target.columns[0]
        train_data, train_target = self.separate_data_and_target(Cursor.TRAIN, target_key)

        # Ax=b linear equation
        A = train_data.T @ train_data
        b = train_data.T @ train_target
        coef = np.linalg.solve(A, b)

        def regressor(): ...
        def predict(data):
            return data @ coef

        regressor.intercept_ = 0
        regressor.coef_ = coef
        regressor.predict = predict

        return regressor


"""
for EOF scroll position
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
--- 
"""
