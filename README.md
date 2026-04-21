# 🧠 제 4회 ETRI 휴먼이해 인공지능 논문경진대회

> **라이프로그 데이터를 활용한 수면 품질 예측을 위한 설명 가능한 시계열 피처 엔지니어링**  
> (데이터는 저작권 문제로 포함되어 있지 않습니다.)

---

## 🏆 대회 개요
- **대회명:** 제 4회 ETRI 휴먼이해 인공지능 논문경진대회  
- **주최:** ETRI (한국전자통신연구원) × Dacon  
- **대회 링크:** https://dacon.io/competitions/open/236468/overview/description  
- **설명:**  
  라이프로그 센서 데이터(활동, 심박 등)를 활용하여 개인의 수면 품질 및 상태(Q1~Q3, S1~S3)를 예측하는 AI 경진대회입니다.
  라이프로그 데이터는 스마트폰 및 웨어러블 센서를 통해 수집된 다중 시계열 데이터로, 이를 활용한 수면 상태 예측은 헬스케어 AI에서 중요한 문제입니다.
  
---

## ⚙️ 실행 환경

| 항목 | 내용 |
|------|------|
| OS | Windows 11 / Google Colab |
| Python | 3.10 |
| 주요 라이브러리 | numpy, pandas, torch, scikit-learn, tqdm, matplotlib, opencv-python |

---

## 📂 데이터

- **데이터 출처:** Dacon 제공
- **형식:**
  - 센서 데이터: `.parquet` (`ch2025_data_items`)
  - 학습 라벨: `ch2025_metrics_train.csv`
  - 제출 샘플: `ch2025_submission_sample.csv`
- **주의:**  
  **데이터는 저작권 문제로 포함하지 않습니다.**

---

## 📁 프로젝트 구조
```bash
ETRI-Lifelog-Sleep-Prediction/
│
├── datasets/
│ ├── val_datasets/
│ │ ├── acc/
│ │ ├── activity/
│ │ ├── hr/
│ │
│ ├── test_datasets/
│ │ ├── acc/
│ │ ├── activity/
│ │ ├── hr/
│ │
│ ├── raw_datasets/
│ └── image_datasets/
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
```bash
[Raw Sensor Data]
↓
[Preprocessing]
↓
[Image Transformation]
↓
[Model Training (CNN)]
↓
[Inference]
↓
[Submission 생성]
```

---

## 🧩 데이터 처리 및 특징 추출

### 🔹 주요 처리 과정

| 구분 | 내용 |
|------|------|
| 센서 데이터 | Activity, Heart Rate, Step 등 |
| 시간 정렬 | 1Hz 기준으로 동기화 |
| 결측치 처리 | interpolation 기반 보간 |
| 정규화 | 센서별 스케일 정규화 |
| 데이터 변환 | 시계열 → 이미지 변환 |

또한 시간 흐름을 유지하기 위해 고정 길이 시퀀스로 변환하고, 센서별 특징이 유지되도록 채널 구조를 구성했습니다.

---

### 🔹 이미지 변환 방식

기존 시계열 모델(RNN, Transformer 등)은 시간 의존성 학습에는 강점이 있지만, 센서 간 상호작용을 직관적으로 학습하는 데 한계가 있습니다.
본 프로젝트에서는 시계열 데이터를 이미지 형태로 변환하여 CNN이 공간 패턴 기반으로 센서 간 관계를 학습할 수 있도록 설계했습니다.

- 시계열 데이터를 시간 축 기준으로 정렬  
- 센서별 값을 행렬 형태로 구성  
- 다채널 이미지로 변환 후 CNN 입력  

---

## 🤖 모델 구조

| 항목 | 내용 |
|------|------|
| 대상 타깃 | Q1~Q3, S1~S3 |
| 모델 | CNN 기반 딥러닝 모델 |
| 구조 | ResNet 계열 |
| 입력 데이터 | 이미지 변환된 시계열 데이터 |
| 학습 방식 | Supervised Learning |
| 평가 지표 | Macro F1-score |

---

### 🔹 Model Variants

| 모델 | 설명 |
|------|------|
| Base | 기본 센서 기반 |
| Full | 확장 피처 포함 |

- **Base:** 빠르고 가벼움  
- **Full:** 성능 향상 (연산량 증가)

---

## 📈 예측 및 결과 생성

- CNN 모델로 수면 상태 예측  
- Softmax 확률 기반 클래스 결정  
- 제출 파일 생성  

---

## 📊 Output
```bash
submission.csv
├── ID
├── Q1, Q2, Q3
└── S1, S2, S3
```

---

## 🚀 실행 방법

```bash
# 1️⃣ 환경 세팅
pip install -r requirements.txt

# 2️⃣ Base 전처리
preprocess_val_base.ipynb 실행
preprocess_test_base.ipynb 실행

# 3️⃣ Full 전처리
preprocess_val_full.ipynb 실행
preprocess_test_full.ipynb 실행

# 4️⃣ Base 모델 학습
train_cnn_base.ipynb 실행

# 5️⃣ Full 모델 학습
train_cnn_full.ipynb 실행

# 6️⃣ Base 추론
inference_cnn_base.ipynb 실행

# 7️⃣ Full 추론
inference_cnn_full.ipynb 실행

# 8️⃣ 제출파일 생성
submission.ipynb 실행
```

---

## 📊 성능
- CNN 기반으로 시계열 패턴 학습
- 센서 간 상호작용을 공간적으로 표현

※ 데이터 미포함으로 인해 정확한 점수 재현은 환경에 따라 달라질 수 있음

---

## 👤 작성자
- **팀명:** 킹운중  

---

## 회고 및 느낀 점
이번 ETRI 라이프로그 기반 수면 건강 예측 AI 경진대회에 참여하며, 단순한 모델 성능 향상을 넘어 데이터와 문제를 바라보는 관점을 깊이 있게 확장할 수 있는 경험을 했습니다.

대회 초반에는 다양한 피처 엔지니어링과 모델 실험에 집중하며 성능 개선을 시도하였으나, 퍼블릭 리더보드 점수를 지속적으로 모니터링하고 전략적으로 활용하지 못한 점이 아쉬움으로 남습니다. 후반에 점수를 확인했을 때 기대보다 낮은 성능을 확인했지만, 제한된 시간 내에 충분한 방향 수정과 검증을 진행하지 못해 최종적으로 상위 30% 수준의 결과를 기록하게 되었습니다.

이번 경험을 통해 모델 자체의 성능뿐만 아니라, 실험 결과를 빠르게 검증하고 방향성을 조정하는 과정, 그리고 리더보드 기반의 전략적 접근이 성능에 큰 영향을 미친다는 점을 체감할 수 있었습니다. 특히 동일한 모델 구조에서도 검증 방식과 실험 설계에 따라 결과가 크게 달라질 수 있다는 점이 인상적이었습니다.

앞으로는 모델링뿐만 아니라 검증 전략, 데이터 분할 방식, 리더보드 활용까지 포함한 전체 파이프라인 관점에서 문제를 접근하고자 합니다. 이번 경험을 통해 얻은 인사이트를 바탕으로, 보다 체계적이고 재현 가능한 실험 환경을 구축하며 지속적으로 성능을 개선해 나가겠습니다.

