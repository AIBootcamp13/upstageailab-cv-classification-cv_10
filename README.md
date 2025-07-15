# Document Classification Competition
## Team
| <img src="https://avatars.githubusercontent.com/u/126853146?s=400&v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/5752438?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/57533441?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/204896949?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/208775216?v=4" width="100"/> |
|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| [오승태](https://github.com/ohseungtae) | [염창환](https://github.com/cat2oon) | [이진식](https://github.com/hoppure) | [안진혁](https://github.com/Quantum-Node-Scott) | [박진섭](https://github.com/seob1504) |
| 팀장<br/>모델링 및<br/>augmentation 실험 | OCR 및<br/>모델링 총괄 | CNN계열 모델<br/>실험 및 성능 비교 | swin transformer<br/>모델링 | EDA 및<br/>일부 모델링 |



## 0. Overview

본 프로젝트는 **Upstage AILAB Computer Vision Class - CV Classification 과제**를 기반으로 진행된 이미지 분류 프로젝트입니다.  
다양한 이미지 증강(Augmentation) 기법과 모델 앙상블 전략을 활용하여, 주어진 이미지 데이터를 17개의 클래스 중 하나로 분류하는 고성능 모델을 구축하는 것을 목표로 하였습니다.

이 리포지토리에는 전체 학습 파이프라인, 데이터 분석(EDA), 전처리 과정, 모델 실험 기록 및 결과가 포함되어 있습니다.

### Environment

- OS: Linux 기반 GPU 서버
- Python: 3.10+
- CUDA: 11.8
- PyTorch: 2.1.0
- GPU: NVIDIA GeForce RTX 3090 

### Requirements

아래의 패키지들이 필요합니다. `requirements.txt`를 통해 설치할 수 있습니다.

pip install -r requirements.txt

## 1. Competiton Info

### Overview

- 대회 주제: 17개 클래스의 문서 타입 이미지 분류
- 도메인: Computer Vision - Document Classification
- 데이터: 현업 실 데이터 기반으로 제작된 문서 이미지 데이터셋
- 목표: 주어진 문서 이미지를 17개 클래스 중 하나로 정확하게 분류하는 모델 개발

### Timeline

### Timeline

| 단계         | 기간                   | 설명                             |
|--------------|------------------------|----------------------------------|
| 데이터 배포   | 2024.06.30 10:00        | 훈련/검증 이미지 및 레이블 제공 |
| 모델 개발     | 2024.06.30 ~ 2025.07.10 | 데이터 분석 및 모델링 진행     |
| 제출 마감     | 2025.07.10             | 모델 결과 제출                  |
| 리뷰 & 발표   | 2025.07.11             | 발표 및 우수 결과 공유          |

## 2. Components

### Directory

```
upstageailab-cv-classification-cv_10/
├── docs/
│   ├── pdf
│   │   └── Upstage AI LAB 4기_10조_발표자료.pdf
├── AJH/      # [ajh] project work
│   ├── swin_base.ipynb # swin transformer 모델
├── chy/     # [chy] project work
│   ├── augs/ # augumentation 실험
│   │   ├── augs.ipynb
│   │   ├── augs.py
│   │   ├── labs.py
│   ├── cnn/ # cnn모델
│   │   ├── augs-b1.py
│   │   ├── augs.py
│   │   ├── cnn-base.ipynb
│   │   ├── ds.py
│   │   ├── labs.py
│   ├── llv3-aug/ # layoutlmv3 코드
│   │   ├── labs.py
│   │   ├── llv3-base.ipynb
│   ├── ocr/ # paddleOCR 진행 코드
│   │   ├── dtc-ocr.ipynb 
│   │   ├── inconsist_list.txt
│   │   ├── inv.py
│   │   ├── labs.py
│   │   ├── ocr-for-test.ipynb
│   ├── vote/ # 지금까지 좋은 성능의 모델들 voting 
│   │   ├── history # 이전 모델 결과물
│   │   │   ├── 보팅용 csv파일들 ...
│   │   ├── probs
│   │   │   ├── cnv2-easy-rot-dice-loss.txt
│   │   │   ├── llv3-dice-loss-sub-classes.txt
│   │   ├── inconsist_list.txt
│   │   ├── inv.py
│   │   ├── labs.py
│   │   ├── vote.ipynb
│   │   ├── vote.py
│   │   ├── voted.csv # 보팅 결과 csv
├── ost/     # [ost] project work
│   ├── Convnext2large/ # Convnext2large모델 관련 코드
│   │   ├── proba-map/
│   │   │   ├── [ost]convnext-proba-map-crop-data.csv # crop data proba map
│   │   │   ├── /[ost]convnext-proba-map-desque-argver2-test.csv #deskew data proba amp
│   │   ├── ConvNextV2-crop-image.ipynb # image 상단부만 crop한 데이터에 대한 모델
│   │   ├── ConvNextV2-deskew.ipynb # deskew된 test data로 학습한 모델
│   │   ├── augs.py
│   │   ├── ds.py
│   │   ├── labs.py
│   ├── augumentation&EDA/ # augmentation 실험 및 data EDA
│   │   ├── augs.ipynb # augmentation 실험 코드
│   │   ├── EDA.ipynb # EDA
│   │   ├── augs.py
│   │   ├── labs.py
│   ├── layoutlmv3/ # layoutlmv3모델 코드
│   │   ├── [ost]llv3-base.ipynb
│   │   ├── augs.py
│   │   ├── labs.py
├── LJS/   # [ljs] project work
│   ├── baseline.ipynb
│   ├── baseline_code.ipynb
│   ├── baselineplversion.ipynb
│   ├── baselineplversionv2.ipynb
│   ├── baselineplversionv4.ipynb
│   ├── baselinev2.ipynb
│   ├── cl-37classifier.ipynb
│   ├── contrastivelearningversion.ipynb
├── tools/                  # [tool] inspector
│   ├── inspector/ # test 데이터 확인을 위한 인스펙터
│   │   ├── doc_classes.json
│   │   ├── meta.ipynb
│   │   ├── tts.csv
│   ├── inspector2/ # test 데이터 확인을 위한 인스펙터 ver2
│   │   ├── inspector.py
│   │   ├── labs.py
│   │   ├── meta.ipynb
│   │   ├── tts.csv
├── .gitignore              
├── LICENSE                 
└── README.md               

```

## 3. Data descrption

### Dataset overview
- **Train**
  - 총 **1,570장**의 문서 이미지로 구성
  - 각 이미지에 대해 **17개 클래스 중 하나**로 라벨링되어 있음
- **Test**
  - 총 **3,140장**의 문서 이미지
  - 라벨은 제공되지 않으며, 최종 예측 제출용
- **Class 종류 (총 17종)**
  - 예시: `진단서`, `여권`, `차량등록증`, `신분증` 등 다양한 문서 포함
- **특징**
  - 클래스 간 **데이터 불균형** 존재
  - **Test 데이터는 Train보다 더 많은 왜곡** (회전, 노이즈, 블러 등) 포함
	
### 학습 데이터 구성

#### `train/`
- 1,570장의 이미지 파일(`.png` 등) 저장됨

#### 📄 `train.csv`
- 각 이미지의 **ID**와 **클래스 라벨 번호** 제공
- 총 1,570행


#### 📄 `meta.csv`
- 클래스 번호(`target`)와 해당 이름(`class_name`) 매핑 정보
- 총 17행



### 평가 데이터 구성

#### `test/`
- 3,140장의 이미지 파일 저장됨 (라벨 없음)

#### 📄 `sample_submission.csv`
- 제출용 샘플 파일
- 총 3,140행 (Test 이미지와 동일 수)
- `target` 값은 전부 0으로 채워져 있음 (예측값 입력 필요)



- 그 밖에 평가 데이터는 학습 데이터와 달리 랜덤하게 Rotation 및 Flip 등이 되었고 훼손된 이미지들이 존재함


### EDA

- **Train 데이터 EDA**
  - **파일 일치 확인**: CSV와 이미지 디렉토리 간 누락된 파일 없음.
  - **클래스 분포 분석**: 상위 14개 클래스는 각 100장으로 균등하지만, 일부 클래스(`resume`, `statement_of_opinion`, `application_for_payment_of_pregnancy_medical_expenses`)는 샘플 수가 적어 불균형 존재.

- **Test 데이터 EDA**
  - **파일 일치 확인**: CSV와 이미지 디렉토리 간 누락된 파일 없음.
  - **해상도 및 비율 분석**: 0.75 (세로형), 1.25 (가로형) 비율 이미지가 대부분을 차지.
  - **밝기 분석**: 대부분 밝은 배경(평균 픽셀 값 180–220), train 대비 밝고 균일함.
  - **마스킹 분석**: 어두운 영역 비율이 매우 낮아 대부분 밝은 문서. (dark ratio 거의 0)
  - **선명도 분석** : test 데이터에 전반적인 높은 변동성의 선명도가 존재함을 확인, 다양한 카메라 장비로 촬영한 것으로 추정됨.
  - **컬러/흑백 비율**: 컬러 이미지와 흑백으로 처리된 이미지가 혼재.
  - **노이즈 정도** : 시각화시 특정 필터를 거쳐야하기에 정확한 수치를 파악할 수 없었으나 인스펙터를 통해 다수의 test 데이터를 직접 확인하여 강한 노이즈와 약한 노이즈가 걸린 이미지를 확인.
  - **회전 정도** : 테스트 데이터에 약한 회전이 걸린 이미지와 강한 회전이 걸린 이미지들이 존재하며, 강한 회전의 경우, 회전시 생기는 가장 자리 손실이 생김을 확인.

### Data Processing

- **데이터 라벨링**
  - Train 데이터는 `train.csv`의 `target` 컬럼을 기준으로 클래스 레이블을 부여.

- **데이터 전처리**
  - Train 이미지의 밝기 및 대비 보정: Test 데이터 분포(밝고 균일)에 맞도록 조정.
  - 회전 및 왜곡 보정: 클래스별 비율 패턴과 회전 상태를 분석해 강한 회정 적용.
  - 노이즈 대비: 약한 노이즈와 강한 노이즈 둘 다 적용하도록 데이터 증강 설계.
  - 흑백 대비 : grey 증강 기법 적용.
  - 마스킹 영역 고려: 클래스별 밝기/어두움 비율을 기반으로 증강 및 보정 전략 설계.
  - 이외에도 데아터를 직접 확인하고 어떤 증강이 적용됬는지 파악후 다양한 데이터 증강 기법 적용.
  - 데이터의 정보를 최대한 활용하기 위해 OCR 적용
  - OCR을 적용하기 전 좋은 정보 추출을 위해 test 데이터 deskew 및 re-orientation 
 
- **데이터 전처리 버전**
  성능 향상을 위해 시간에 따른 적용한 전처리가 다름
  1. augrapy와 albumentation을 동시에 적용한 강한 data augumentation 적용
  2. resize_and_crops, jitters, colors, rotators 등으로 albumentation의 분류를 나눈 다음, radom select를 통해 100개의 경우의 수를 만들고, 모델의 에포크마다 다르게 적용
  3.  test 데이터 deskew 작업 후, 이미 회전과 반전에 대해 원래 이미지로 거의 복구했으니 약한 회전과 훼손이나 노이즈 같은 augmentation을 적용 



## 4. Modeling

### Model Description

본 프로젝트에서는 **Layoutlmv3, Convnext 계열 모델**을 주력으로 사용하여 문서 분류 성능을 극대화했습니다.

#### 사용된 모델 아키텍처
(여기에 사용한 모델 더 넣어주길 바람)
- **ResNet-34**: 초기 베이스라인 모델로 사용
- 
- **ConvNeXtV2-Base**: 개선된 ConvNeXt 아키텍처의 기본 모델
- **ConvNeXtV2-Large**: 개선된 ConvNeXt 아키텍처의 대용량 모델
- **Swin transformer**: Shifted Window Attention기반의 VIT 아키텍처의 모델
- **Layoutlmv3** : 텍스트, 레이아웃, 이미지 정보를 동시에 처리할 수 있는 모델

#### 모델 선택 이유

1. **높은 성능**: ImageNet에서 검증된 SOTA 성능
2. **효율성**: 파라미터 대비 높은 성능 효율
3. **전이학습 적합성**: 사전 훈련된 가중치를 활용한 빠른 학습
4. **문서 이미지 특성**: 세밀한 텍스트와 구조 인식에 우수한 성능

### Modeling Process

#### 1. 모델링 아키텍처
- Pytorch Lightning 기반으로 재사용 가능한 Module 구성
- 384x384의 고해상도 입력
- WandB를 통한 실험 기록

 #### 2.메트릭 시스템 구축
 - 전체 정확도를 뜻하는 F1 score
 - 모델별 학습 - **컬러/흑백 비율**: 컬러 이미지와 흑백으로 처리된 이미지가 혼재.
  - **노이즈 정도** : 시각화시 특정 필터를 거쳐야하기에 정확한 수치를 파악할 수 없었으나 인스펙터를 통해 다수의 test 데이터를 직접 확인하여 강한 노이즈와 약한 노이즈가 걸린 이미지를 확인.
  - **회전 정도** : 테스트 데이터에 약한 회전이 걸린 이미지와 강한 회전이 걸린 이미지들이 존재하며, 강한 회전의 경우, 회전시 생기는 가장 자리 손실이 생김을 확인.

### Data Processing

- **데이터 라벨링**
  - Train 데이터는 `train.csv`의 `target` 컬럼을 기준으로 클래스 레이블을 부여.

- **데이터 전처리**
  - Train 이미지의 밝기 및 대비 보정: Test 데이터 분포(밝고 균일)에 맞도록 조정.
  - 회전 및 왜곡 보정: 클래스별 비율 패턴과 회전 상태를 분석해 강한 회정 적용.
  - 노이즈 대비: 약한 노이즈와 강한 노이즈 둘 다 적용하도록 데이터 증강 설계.
  - 흑백 대비 : grey 증강 기법 적용.
  - 마스킹 영역 고려: 클래스별 밝기/어두움 비율을 기반으로 증강 및 보정 전략 설계.
  - 이외에도 데아터를 직접 확인하고 어떤 증강이 적용됬는지 파악후 다양한 데이터 증강 기법 적용.
  - OCR을 적용하기 전 좋은 정보 추출을 위해 test 데이터 deskew 및 re-orientation, flip-recover
  - 데이터의 정보를 최대한 활용하기 위해 OCR 적용
 
- **데이터 전처리 버전**
  성능 향상을 위해 모델에 따른 적용한 전처리가 다름
  1. augrapy와 albumentation을 통한 강한 augumentation을 확률적으로 적용
  2. resize_and_crops, jitters, colors, rotators 등으로 albumentation의 분류를 나눈 다음, radom select를 통해 100개의 경우의 수를 만들고, 모델의 에포크마다 다르게 적용
  3. test 데이터 deskew 작업 후, 이미 회전과 반전에 대해 원래 이미지로 거의 복구했으니 약한 회전과 훼손이나 노이즈 같은 augmentation을 적용 



## 4. Modeling

### Model Description

본 프로젝트에서는 **Layoutlmv3, Convnext 계열 모델**을 주력으로 사용하여 문서 분류 성능을 극대화했습니다.

#### 사용된 모델 아키텍처
(여기에 사용한 모델 더 넣어주길 바람)
- **ResNet-34**: 초기 베이스라인 모델로 사용
- 
- **ConvNeXtV2-Base**: 개선된 ConvNeXt 아키텍처의 기본 모델
- **ConvNeXtV2-Large**: 개선된 ConvNeXt 아키텍처의 대용량 모델
- **Swin transformer**: Shifted Window Attention기반의 VIT 아키텍처의 모델
- **Layoutlmv3** : 텍스트, 레이아웃, 이미지 정보를 동시에 처리할 수 있는 모델

#### 모델 선택 이유

1. **높은 성능**: ImageNet에서 검증된 SOTA 성능
2. **효율성**: 파라미터 대비 높은 성능 효율
3. **전이학습 적합성**: 사전 훈련된 가중치를 활용한 빠른 학습
4. **문서 이미지 특성**: 세밀한 텍스트와 구조 인식에 우수한 성능

### Modeling Process

#### 1. 모델링 아키텍처
- Pytorch Lightning 기반으로 재사용 가능한 Module 구성
- 384x384의 고해상도 입력
- WandB를 통한 실험 기록
- Transfer Learning: ImageNet-22k 사전훈련으로 일반화 성능 향상

#### 2. 정교한 메트릭 시스템 구축
 - 정확도, f1 score, roc-auc 등 다양한 score 사용 
 - 모델의 클래스별 학습 정확도를 뜻하는 per-class-accuracy도 추가하여 각 모델마다 잘 예측하는 클래스를 찾아냄

#### 3. 훈련 과정 (Training Process)
- 배치 단위 처리: 효율적인 GPU 메모리 사용
- 실시간 메트릭 업데이트: 각 배치마다 성능 지표 갱신
- 로그 기록: 학습률, 손실값, 각종 메트릭 실시간 모니터링

#### 4. 최적화 전략
- **복합 학습률 스케줄링**
  - Delay Phase (0-30 steps): 초기 가중치 안정화
  - Warmup Phase (30-40 steps): 점진적 학습률 증가
  - Main Phase (40+ steps): 코사인 어닐링으로 부드러운 감소
- **옵티마이저**
  - AdamW: 가중치 감쇠를 포함한 Adam 최적화
  - 학습률 1e-5: 사전훈련 모델에 적합한 낮은 학습률
  - Weight Decay 1e-4: 과적합 방지
- **조기 종료**: 검증 성능 기준 조기 종료
- **모델 저장**: 최고 F1 점수 기준 모델 저장

#### 5. 검증 과정 (Validation Process)
- **성능 모니터링**
  - 에포크별 종합 평가: 전체 검증 데이터셋에 대한 성능 계산
  - 클래스별 세부 분석: 각 문서 타입별 성능 추적
  - 실시간 모니터링: WandB를 통한 시각화
  
#### 6. 훈련 설정 (Training Configuration)
- **Mixed Precision**: 메모리 사용량 50% 감소, 속도 향상
- **ModelCheckpoint**: valid_loss 기준 최적 모델 자동 저장
- **DataLoader Reload**: 매 에포크마다 새로운 augmentation 적용

## 5. voting

### 예측 결과 분석
<img width="1600" height="512" alt="1" src="https://github.com/user-attachments/assets/4341cf9b-e157-4b91-b406-1c27709c1c33" />
- per-class-accuracy를 통해 각 모델이 예측하는 클래스 정확도를 알 수 있음
- 이 결과를 통해 아래의 인사이트를 도출함 

#### Classification Performance by Document Type

| Performance Level | Class ID | Document Type | Notes |
|-------------------|----------|---------------|-------|
| **Excellent** | 0 | 계좌번호 | |
| **Excellent** | 2 | 차량 계기판 | |
| **Excellent** | 5 | 운전면허증 | |
| **Excellent** | 8 | 민증 | |
| **Excellent** | 9 | 여권 | |
| **Excellent** | 15 | 차량등록증 | |
| **Excellent** | 16 | 자동차 | |
| **Good** | 6 | 의료비 영수증 | |
| **Good** | 10 | 납입 확인서 | |
| **Good** | 11 | 약국 영수증 | |
| **Good** | 12 | 처방전 | |
| **Poor** | 1 | 임신/출산 | |
| **Poor** | 3 | 입퇴원 확인서 | Often confused with 외래 진료 확인서 |
| **Poor** | 4 | 진단서 | Often confused with 소견서 |
| **Poor** | 7 | 외래 진료/통원 확인서 | Often confused with 입퇴원 확인서 |
| **Poor** | 13 | 이력서 | |
| **Poor** | 14 | 소견서 | Often confused with 진단서 |

- 이를 통해 CNN계열 모델들이 완벽히 맞추는 클래스를 찾아냄
- 계좌번호(0), 차량 계기판(2), 운전면허증 (5), 민증(8), 여권(9), 차량등록증(15), 자동차계기판(16) wandb상 완벽히 맞추는 것을 확인함
- 이를 통해 특별한 보팅 전략을 고안함

### 특별한 보팅 전략 고안

#### 초기 전략
- 계좌번호(0), 차량 계기판(2), 운전면허증 (5), 민증(8), 여권(9), 차량등록증(15), 자동차계기판(16) 
  easy 7종은 CNN 모델로 바로 확정

- 그 후 OCR 텍스트 정보의 강점을 살리기 위해 LayoutLMv3와 Text Classification 모델 훈련 적용
  - 2~3개 클래스 혹은 남은 10 클래스 등으로 훈련하였으나 최종적으로는 10 클래스 훈련
 
#### 최종 전략

- **1단계 (1400개)** :  easy 7종에 대해 가장 성능이 좋았던 ConvNextV2-384로 바로 확정 (1400개 확정)
- **2단계 (2732개)** : 멤버들이 훈련했던 각 모델들 결과 중 90% 일치하는 항목들로 선정 (1332개 확정)
- **3단계 (3051개 )**" 텍스트 분류 모델로 예측 엔트로피 값이 0.7보다 낮은 항목에서 soft voting (319개 확정)
    - 텍스트 분류기는 약국 영수증과 의료비 영수증 혼동이 커서 두개 클래스는 추출에서 제외
- **4단계 (3140개)** :  남은 89개 항목은 conv 계열 중에서 가장 성능이 좋았던 4 모델에 대해 다수결 하드 보팅


## 6. Result

### Leader Board
<img width="1222" height="477" alt="image" src="https://github.com/user-attachments/assets/753a5661-b1cd-4888-aeb8-d7978fa2ace1" />
<img width="1230" height="485" alt="image" src="https://github.com/user-attachments/assets/a4996f83-a0ba-4a7b-8443-6392bffae275" />


#### 최종 성능 결과

- **최고 성능 모델**: voting model
- **리더보드**
  - mid score: 0.9799
  - final score: 0.9694
- **모델 구성**: Convnext2large + layoutlmv3 + swin transformer + text classifier


#### 실험 결과 요약

| 모델 | 이미지 크기 | 전처리버전  | 검증 F1 | 리더보드 |
|------|-------------|---------|---------|----------|


| ConvnextV2-large | 384×384 | ver3 | 0.9291 | **0.9396** |

### Presentation

- doc폴더내에 공유함

## etc

### Meeting Log

- 팀 회의록은 내부 협업 도구인 Linear를 통해 관리
- 주요 결정사항과 실험 결과는 코드 내 주석 및 로그 파일로 관리




