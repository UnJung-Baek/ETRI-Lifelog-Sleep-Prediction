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

라이프로그 데이터는 센서별 수집 주기와 결측 구조가 서로 다르기 때문에,  
시간축을 기준으로 데이터를 정렬하고 정합성을 맞추는 과정이 필수적입니다.  

특히 결측 구간을 단순 제거할 경우 중요한 패턴이 손실될 수 있기 때문에,  
연속적인 시간 흐름을 유지하기 위해 interpolation 보간을 적용했습니다.

또한 시계열을 직접 시각화하며 이상 구간을 확인하고,  
노이즈와 실제 패턴을 구분하여 데이터 품질을 정제하는 작업도 실시했습니다.  


---

### 🔹 시계열 → 이미지 변환

본 프로젝트에서는 기존의 시계열 모델(RNN, Transformer)이 아닌  
CNN 기반 모델을 활용하기 위해 시계열 데이터를 이미지로 변환했습니다.

#### 이미지로 변환 이유

라이프로그 데이터는 단일 시계열이 아니라  
여러 센서(Activity, HR 등)가 동시에 기록되는 다변량 시계열 데이터입니다.

기존 시계열 모델은 시간 의존성은 잘 학습하지만, 센서 간 상호작용을 명시적으로 학습하기 어렵습니다.

이를 해결하기 위해 다음과 같은 접근을 사용했습니다:

- 시간 축을 기준으로 시계열을 정렬
- 센서별 값을 행(row)으로 배치
- 시간 흐름을 열(column)로 구성
- 다채널 이미지로 변환

이 변환을 통해 모델은 다음을 학습하게 됩니다:

- 세로 방향: 센서 간 관계 (HR ↔ Activity 등)
- 가로 방향: 시간에 따른 변화 패턴
- 전체 패턴: 수면 상태와 관련된 복합적 구조

즉, CNN은 단순 시계열이 아니라  
“센서 × 시간” 구조의 공간 패턴을 학습하게 됩니다.

결과적으로 센서 간 상호작용 + 시간 패턴을 동시에 학습 가능하게 됩니다.

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

CNN 기반 구조를 선택한 이유는  
이미지로 변환된 시계열 데이터에서 공간 패턴 학습에 강점을 가지기 때문입니다.

특히 ResNeXt 계열 모델은  
다양한 feature를 병렬적으로 학습하는 구조를 가지고 있어  
복잡한 센서 패턴을 효과적으로 표현할 수 있습니다.

---

### 🔹 Model Variants

| 모델 | 설명 |
|------|------|
| Base | 기본 센서 기반 |
| Full | 확장 센서 포함 |

- Base: 핵심 센서만 사용 → 안정적인 패턴 학습
- Full: 전체 센서 사용 → 추가 정보 활용

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

