# Document Classification Competition
## Team
| <img src="https://avatars.githubusercontent.com/u/126853146?s=400&v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/5752438?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/57533441?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/204896949?v=4" width="100"/> | <img src="https://avatars.githubusercontent.com/u/208775216?v=4" width="100"/> |
|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
| [오승태](https://github.com/ohseungtae) | [염창환](https://github.com/cat2oon) | [이진식](https://github.com/hoppure) | [안진혁](https://github.com/Quantum-Node-Scott) | [박진섭](https://github.com/seob1504) |
| 팀장<br/>모델링 및<br/>augmentation 실험 | 모델링 및<br/>전처리 총괄 | CNN계열 모델<br/>실험 및 성능 비교 | swin transformer<br/>모델링 | EDA 및<br/>일부 모델링 |



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
  - **밝기 분석**: 대부분 밝은 배경(평균 픽셀 값 180–220), train 대비 훨씬 밝고 균일함.
  - **마스킹 분석**: 어두운 영역 비율이 매우 낮아 대부분 밝은 문서. (dark ratio 거의 0)
  - **컬러/흑백 비율**: 컬러 이미지와 흑백으로 처리된 이미지가 혼재

### Data Processing

- **데이터 라벨링**
  - Train 데이터는 `train.csv`의 `target` 컬럼을 기준으로 클래스 레이블을 부여.

- **데이터 클리닝 및 전처리**
  - Train 이미지의 밝기 및 대비 보정: Test 데이터 분포(밝고 균일)에 맞도록 조정.
  - 회전 및 왜곡 보정: 클래스별 비율 패턴과 회전 상태를 분석해 강한 회정 적용.
  - 노이즈 제거 및 텍스트 강화: 보안 문서류는 강한 노이즈 제거 및 에지 강화, 의료/금융 문서는 부드러운 노이즈 제거 및 선명도 향상.
  - 마스킹 영역 고려: 클래스별 밝기/어두움 비율을 기반으로 증강 및 보정 전략 설계.

- **클래스 불균형 대응**
  - 클래스 가중치 조정 및 소수 클래스 중심 데이터 증강 전략 적용.


## 4. Modeling

### Model Description

본 프로젝트에서는 **Layoutlmv3, Convnext 계열 모델**을 주력으로 사용하여 문서 분류 성능을 극대화했습니다.

#### 사용된 모델 아키텍처

- **ResNet-34**: 초기 베이스라인 모델로 사용
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

#### 1. 베이스라인 모델 (ResNet34)
....



#### 4. 데이터 증강 
- **albumentation**: 
  - 회전 (±10도)
  - 밝기/대비 조정
  - 가우시안 노이즈 추가
  - 크기 조정 및 크롭
- **augrapy**: 
  - 회전 (±10도)
  - 밝기/대비 조정
  - 가우시안 노이즈 추가
  - 크기 조정 및 크롭
#### 5. 최적화 전략
- **손실 함수**: CrossEntropyLoss
- **옵티마이저**: Adam (학습률 0.001)
- **스케줄러**: CosineAnnealingLR
- **조기 종료**: 검증 성능 기준 조기 종료
- **모델 저장**: 최고 F1 점수 기준 모델 저장

#### 6. 성능 향상 기법


## 5. Result

### Leader Board

#### 최종 성능 결과

- **최고 성능 모델**: ConvnextV2-Base + 10x Aug + TTA
- **검증 성능**: 
  - F1 Score: 
  - 리더보드: 0.9418
- **모델 구성**: 


#### 실험 결과 요약

| 모델 | 이미지 크기 | 전처리  | 검증 F1 | 리더보드 |
|------|-------------|---------|---------|----------|
...

### Presentation

- 발표 자료는 프로젝트 완료 후 업데이트 예정

## etc

### Meeting Log

- 팀 회의록은 내부 협업 도구인 Linear를 통해 관리
- 주요 결정사항과 실험 결과는 코드 내 주석 및 로그 파일로 관리

### Reference

#### 주요 참고 문헌


