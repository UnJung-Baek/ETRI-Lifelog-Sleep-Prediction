# 🧠 제 4회 ETRI 휴먼이해 인공지능 논문경진대회

라이프로그 시계열 데이터를 이미지 기반 표현으로 변환하여 CNN 기반 수면 상태 예측 모델 개발
(데이터는 저작권 문제로 포함되어 있지 않습니다.)

---

## 🏆 대회 개요

- 대회명: 제 4회 ETRI 휴먼이해 인공지능 논문경진대회  
- 주최: ETRI (한국전자통신연구원) × Dacon  
- 대회 링크: https://dacon.io/competitions/open/236468/overview/description  

라이프로그 센서 데이터(활동, 심박 등)를 활용하여  
개인의 수면 상태 및 설문 지표(Q1~Q3, S1~S3)를 예측하는 문제입니다.

본 대회는 다중 센서 기반 시계열 데이터를 활용하여  
인간의 행동 및 건강 상태를 이해하는 헬스케어 AI 문제를 다룹니다.

---

## ⚙️ 실행 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 / Google Colab |
| Python | 3.10 |
| 주요 라이브러리 | numpy, pandas, torch, timm, albumentations, scikit-learn |

---

## 📂 데이터

- 형식:
  - 센서 데이터: `.parquet` (Activity, HR, Step 등)
  - 학습 라벨: `ch2025_metrics_train.csv`
  - 제출 샘플: `ch2025_submission_sample.csv`

※ 데이터는 저작권 문제로 포함하지 않습니다.

---

## 📁 프로젝트 구조

```bash
ETRI-Lifelog-Sleep-Prediction/
│
├── datasets/
│   ├── val_datasets/
│   ├── test_datasets/
│   ├── raw_datasets/
│   └── image_datasets/
│
├── preprocess_val_base.ipynb
├── preprocess_val_full.ipynb
├── preprocess_test_base.ipynb
├── preprocess_test_full.ipynb
│
├── train_cnn_base.ipynb
├── train_cnn_full.ipynb
│
├── inference_cnn_base.ipynb
├── inference_cnn_full.ipynb
│
├── submission.ipynb
└── README.md
```

---

## 🔄 전체 파이프라인

본 프로젝트는 라이프로그 센서 시계열 데이터를 이미지 기반 표현으로 변환하여  
CNN 모델을 통해 수면 상태를 예측하는 구조로 구성되어 있습니다.

```bash
Raw Sensor Data (.parquet)
↓
Preprocessing
  - 센서 데이터 통합 (Activity, HR, Step 등)
  - timestamp 정렬 및 중복 제거
  - 결측치 보간 (interpolation)
  - 사용자-날짜 단위 시계열 구성
↓
Image Transformation
  - 시계열 데이터를 행렬 형태로 변환
  - 센서별 채널 구성
  - CNN 입력용 이미지 생성
↓
Model Training (CNN)
  - Base 모델 (기본 센서)
  - Full 모델 (확장 센서)
↓
Inference
  - Base 모델 예측
  - Full 모델 예측
↓
Ensemble
  - Base + Full 결과 결합
  - Hard Voting 기반 최종 예측 생성
↓
Submission
  - submission.csv 생성
```

---

## 🧩 데이터 처리 및 특징 추출

### 🔹 주요 처리 과정

- Activity, Heart Rate, Step 등 다중 센서 데이터 통합
- timestamp 기준 정렬 및 중복 데이터 제거
- 결측치에 대해 interpolation 기반 보간 수행
- 센서별 값 스케일 정규화
- 사용자-날짜 단위로 시계열 재구성

단순한 전처리를 넘어, 시계열을 직접 시각화하며 이상 구간과 실제 패턴을 구분하여  
데이터 품질을 정제하는 과정을 포함했습니다.

---

### 🔹 시계열 → 이미지 변환

본 프로젝트에서는 시계열 데이터를 CNN에 활용하기 위해 이미지 형태로 변환했습니다.

- 시간 축 기준으로 정렬된 시계열 구성  
- 센서별 값을 행렬 형태로 변환  
- 다채널 이미지로 구성하여 CNN 입력  

이를 통해 시간 흐름뿐 아니라  
센서 간 상호작용을 공간 패턴으로 학습할 수 있도록 설계했습니다.

---

## 🤖 모델 구조

| 항목 | 내용 |
|------|------|
| 문제 유형 | 멀티라벨 분류 |
| 타깃 | Q1~Q3, S1~S3 |
| 모델 | CNN 기반 (ResNeXt / SE-ResNeXt) |
| 입력 | 이미지 변환된 시계열 데이터 |
| Loss | BCE Loss |
| 평가 지표 | Macro F1-score |

---

### 🔹 Model Variants

| 모델 | 설명 |
|------|------|
| Base | 기본 센서 기반 |
| Full | 확장 센서 포함 |

- Base: 주요 센서만 사용하여 안정적인 학습  
- Full: 더 많은 센서를 포함하여 정보량 확장 및 성능 개선  

---

## 📈 예측 및 앙상블

각 모델의 예측 결과를 결합하여 최종 결과를 생성했습니다.

- Base 모델 예측
- Full 모델 예측

두 결과를 결합하여 다음과 같은 방식으로 최종 예측을 생성했습니다.
Base 결과 + Full 결과 → Hard Voting → 최종 예측

---

## 🚀 실행 방법

1. 전처리
   - `preprocess_val_base.ipynb` 실행
   - `preprocess_test_base.ipynb` 실행
   - `preprocess_val_full.ipynb` 실행
   - `preprocess_test_full.ipynb` 실행

2. 모델 학습
   - `train_cnn_base.ipynb` 실행
   - `train_cnn_full.ipynb` 실행

3. 예측
   - `inference_cnn_base.ipynb` 실행
   - `inference_cnn_full.ipynb` 실행

4. 제출 생성
   - `submission.ipynb` 실행

---

## 📊 성능 및 특징

- 시계열 데이터를 이미지로 변환하여 CNN 기반 학습 적용
- 센서 간 관계를 공간 패턴으로 학습
- Base / Full 모델을 결합한 앙상블 구조

---

## 👤 작성자

- **팀명:** 킹운중

---

## 🏅 성과

- 최종 상위 30% 기록
- 시계열 → 이미지 변환 기반 CNN 모델링 적용
- 다중 센서 데이터 통합 및 전처리 파이프라인 구축
- Base / Full 모델 앙상블을 통한 예측 안정성 확보

---

## 📝 느낀점

이번 프로젝트를 통해 가장 크게 느낀 점은  
모델보다 데이터 전처리와 표현 방식이 성능에 더 큰 영향을 미친다는 점이었습니다.

특히 라이프로그 시계열 데이터를 단순 입력으로 사용하는 것이 아니라,  
이미지 형태로 변환하여 CNN이 센서 간 관계를 학습하도록 한 접근이  
성능 개선에 중요한 역할을 했습니다.

또한 Base와 Full 모델을 분리하여 학습하고  
이를 앙상블하는 구조를 통해 예측의 안정성을 확보할 수 있었습니다.

다만 이번 프로젝트에서는 OOF 기반 검증에만 의존하고  
퍼블릭 리더보드 점수를 충분히 확인하며 전략적으로 활용하지 못한 점이 아쉬움으로 남습니다.  
그 결과 모델 성능 대비 최종 점수를 충분히 끌어올리지 못했고,  
최종적으로 상위 30%를 기록하게 되었습니다.

이번 경험을 통해 모델링뿐만 아니라  
검증 방식, 데이터 분할 전략, 그리고 리더보드 활용까지 포함한  
전체 파이프라인 관점에서 접근하는 것이 중요하다는 점을 느꼈습니다.

