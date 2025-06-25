# InternVL3 HIVAU-70k Fine-tuning & Debugging & eval Guide

## ⚙️ 환경 설정 (Prepare the Environment)

InternVL 모델을 로컬에서 실행하거나 디버깅하기 위해 먼저 Python 환경을 구성합니다.

```bash
conda create -n vl3 python=3.9
conda activate vl3

pip install -r requirements.txt

# option: Chat 모델 학습에 최적화된 FlashAttention 설치
pip install flash-attn==2.3.6 --no-build-isolation
```


## 📂 데이터셋 구성 안내

본 프로젝트는 아래와 같은 폴더 구조가 이미 존재한다는 가정하에 작성되었습니다. 
실험을 위해서는 동일한 디렉토리 구조로 데이터셋을 구성해야 합니다:

```
assets/
internvl_chat/
...
HIVAU-70k/
├── instruction/
├── raw_annotations/
└── videos/
    ├── ucf-crime/
    │   ├── clips/
    │   │   ├── test/
    │   │   └── train/
    │   ├── events/
    │   │   ├── test/
    │   │   └── train/
    │   └── videos/
    │       └── train/
    └── xd-violence/
        ├── clips/
        │   ├── test/
        │   └── train/
        ├── events/
        │   ├── test/
        │   └── train/
        └── videos/
            ├── test/
            └── train/

```

> ⚠️ 위와 같은 구조가 맞지 않으면 학습 및 평가 코드가 정상 동작하지 않을 수 있습니다.

해당 데이터셋은 아래 GitHub 저장소에서 확인 및 다운로드할 수 있습니다:
🔗 [https://github.com/jungseoik/HIVAU-70k](https://github.com/jungseoik/HIVAU-70k)
구축 시 전체 압축을 풀고 반드시 위 폴더 구조에 맞게 정리해 주세요.


## InternVL3-2B 모델 다운로드 (프로젝트 ckpts 폴더에 저장)

이 프로젝트는 Hugging Face에서 공개된 [OpenGVLab/InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) 모델을 사용합니다. 
아래 명령어를 통해 사전에 모델을 다운로드해두어야 파인튜닝이 가능합니다.

### 다운로드 방법

```bash
# 1. ckpts 디렉토리 생성
mkdir -p ckpts

# 2. InternVL3-2B 모델 다운로드
huggingface-cli download \
  --resume-download \
  --local-dir-use-symlinks False \
  OpenGVLab/InternVL3-2B \
  --local-dir ckpts/InternVL3-2B
```

> ⚠️ 위 명령을 실행하려면 먼저 `huggingface-cli login` 명령어로 Hugging Face 계정에 로그인되어 있어야 합니다.

---

### 📁 다운로드 후 디렉토리 구조 예시

```plaintext
ckpts/
└── InternVL3-2B/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── generation_config.json
    └── ... 
```



## 🚀 실행 방식 선택

InternVL 파인튜닝은 다음 두 가지 방식 중 하나로 실행할 수 있습니다:

---

### 1. 쉘 스크립트를 통한 대규모 학습 실행

```bash
# 8개의 GPU를 사용하여 전체 LLM 파인튜닝 (GPU당 약 30GB 메모리 사용)
cd internvl_chat
GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl3.0/2nd_finetune/internvl3_2b_finetune_lora_custom.sh
```

### 2. VSCode 디버깅 환경에서 실행

위 항목들을 수정하여 VSCode의 `launch.json`으로 디버깅 실행하면,
- 특정 파라미터 실험이나 오류 추적이 쉽습니다.(다만 적합한 실행방식은 아닙니다. 흐름을 알고 싶다면 체크하는건 추천합니다)

> 아래 방법은 개발/디버깅 단계에 적합합니다.

#### 🛠 디버깅을 따로 하고 싶다면 본인 환경에서 수정해야 할 항목들

InternVL 파인튜닝을 다른 환경에서 디버깅하려면 'assets/launch.json" 내용에 해당하는 아래 항목들을 **자신의 시스템에 맞게 수정**해야 합니다.

#### 🔧 필수 수정 대상 (환경 종속성 있음)

| 구분         | 항목                    | 설명                              | 예시 경로 또는 값 |
|--------------|-------------------------|-----------------------------------|-------------------|
| 📁 경로 관련 | `"program"`             | 학습 스크립트 경로                | `internvl_chat/internvl/train/internvl_chat_finetune.py` |
|              | `--model_name_or_path`  | 사전 학습 모델 경로 (로컬)        | `/home/USER/.../ckpts/InternVL3-2B` |
|              | `--output_dir`          | 학습 결과 저장 경로               | `/home/USER/.../ckpts/output_lora` |
|              | `--meta_path`           | 학습용 데이터 JSON 경로           | `/home/USER/.../data/custom_data.json` |
|              | `--deepspeed`           | Deepspeed 설정 파일 경로          | `${workspaceFolder}/internvl_chat/zero_stage1_config.json` |
|              | `PYTHONPATH`            | 내부 모듈 import를 위한 경로 설정 | `${workspaceFolder}/internvl_chat` |

---

| 구분         | 항목              | 설명                                 | 예시 경로 또는 값 |
|--------------|-------------------|--------------------------------------|-------------------|
| ⚙️ CUDA 설정 | `CUDA_HOME`       | 설치된 CUDA 경로                     | `/usr/local/cuda-12.3` |
|              | `PATH`            | CUDA 실행 파일이 포함된 경로 추가    | `/usr/local/cuda-12.3/bin:${env:PATH}` |
|              | `LD_LIBRARY_PATH` | CUDA 라이브러리 경로 추가            | `/usr/local/cuda-12.3/lib64:${env:LD_LIBRARY_PATH}` |

---
