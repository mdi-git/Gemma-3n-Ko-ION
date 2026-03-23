import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

MODEL_ID = "openai/gpt-oss-120b"
TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "./outputs/gpt-oss-120b-qlora"

# H100 80GB 1장 기준 시작점
MAX_LENGTH = 2048
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # gpt-oss는 chat template를 쓰면 harmony format이 자동 적용됨
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA용 4-bit NF4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",  # flash-attn 환경이면 바꿔도 됨
    )

    model.config.use_cache = False

    # 먼저 attention 위주로 시작
    # OpenAI 한국어 예제도 attention-only부터 시작하고,
    # 필요 시 MoE expert projection까지 확장하라고 안내함
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

    # JSONL 로드
    ds = load_dataset("json", data_files={"train": TRAIN_FILE})

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = ds["train"].map(format_example, remove_columns=ds["train"].column_names)

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
        save_steps=200,
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

