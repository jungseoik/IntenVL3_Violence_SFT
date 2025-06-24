
# 📊 InternVL3 Violence Classification Evaluator

이 프로젝트는 비디오 기반 폭력 분류 모델을 평가하고, 결과 리포트를 생성하는 실험용 파이프라인입니다.

---

## 📁 프로젝트 구조 요약

```plaintext
├── evaluator/
│   └── eval_cls_vid.py           # 결과 CSV로부터 리포트 생성
├── extractor/
│   └── ucf_video.py              # 모델 평가 로직 (eval 함수 포함)
├── assets/
│   └── config.py                 # 설정값들 (모델 리스트, 템플릿 등)
├── results.csv                   # 평가 결과 CSV 파일 (자동 생성됨)
└── run.py                        # 이 스크립트를 실행하세요
```

---

## ⚙️ 설치 및 환경 구성

Python ≥ 3.8 권장

```bash
conda create -n violence_eval python=3.9 -y
conda activate violence_eval
pip install -r requirements.txt

```

---

## 🚀 실행 방법

### 🔹 1. 평가 수행

모델, 템플릿, 세그먼트 수 조합별로 비디오 분류 평가를 수행합니다.

```bash
python run.py --mode eval
```

> 결과는 `results.csv`로 저장됩니다 (경로는 `assets/config.py`의 `OUTPUT_CSV` 참고).

---

### 🔹 2. 리포트 생성

평가 결과 CSV를 기반으로 정리된 리포트를 생성합니다.

```bash
python run.py --mode report --csv results.csv
```

> `--csv` 경로는 생략 시 기본값 `"results.csv"`가 사용됩니다.

---

## 🧩 구성 요소 설명

* `eval()`: 모든 모델/템플릿/세그먼트 조합에 대해 비디오 분류를 실행합니다.
* `generate_comprehensive_report(csv_path)`: 평가 결과 CSV를 분석하여 종합적인 리포트를 생성합니다.
* `InternVL3Inferencer`: 실제 모델 추론 로직을 담당합니다.

---

## 📝 설정 변경

`assets/config.py`를 열어 아래 항목을 수정할 수 있습니다:

* `VIDEO_FOLDER`: 평가할 비디오 폴더 경로
* `VIDEO_CATEGORIES_FILE`: ground-truth 레이블 JSON
* `MODEL_LIST`, `TEMPLATES`, `NUM_SEGMENTS_LIST`: 평가할 조건 조합
* `MAX_WORKERS`: 병렬 처리 개수

---

## 예시

```bash
# 모든 실험 조합에 대해 평가 수행
python run.py --mode eval

# 결과 분석 리포트 생성
python run.py --mode report --csv results.csv
```

---

## 결과 예시 (CSV 포맷)

| video\_name | ground\_truth | model\_name | template\_type | predicted\_category | num\_segment |
| ----------- | ------------- | ----------- | -------------- | ------------------- | ------------ |
| fight1.mp4  | Violence      | internvl-v1 | typeA          | Violence            | 8            |
| normal1.mp4 | NonViolence   | internvl-v2 | typeB          | NonViolence         | 16           |

---

