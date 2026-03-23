import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "openai/gpt-oss-120b"
TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "./outputs/gpt-oss-120b-lora"

MAX_LENGTH = 1024
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 핵심: BitsAndBytesConfig 제거
    # gpt-oss 기본 MXFP4 모델을 학습용으로 dequantize 해서 로드
    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,          # torch_dtype 대신 dtype
        attn_implementation="eager",   # OpenAI 예제는 eager 사용
        device_map="auto",
        use_cache=False,
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": TRAIN_FILE})

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = ds["train"].map(
        format_example,
        remove_columns=ds["train"].column_names
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_length=MAX_LENGTH,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
