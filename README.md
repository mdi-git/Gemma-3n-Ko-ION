# Gemma-3n-Ko-ION

한국어 LLM 학습 및 데이터 생성용 스크립트 모음.

## 포함 파일

- `train_llama31_8b_lora.py`
  - Llama 3.1 8B Instruct LoRA 학습 스크립트

- `make_big_train_jsonl.py`
  - 대규모 SFT용 JSONL 데이터 생성 스크립트

- `make_sft_from_numeric_csv.py`
  - 대형 수치 CSV를 시계열 설명형 SFT 데이터로 변환하는 스크립트

## 설치

```bash
pip install -r requirements.txt
