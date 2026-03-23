import os
import json
import argparse
from typing import List, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Mxfp4Config,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


class NoMoveSFTTrainer(SFTTrainer):
    """
    device_map='auto' / offload 로 이미 배치된 모델에 대해
    Trainer가 다시 model.to(device) 하지 않도록 막는다.
    """
    def _move_model_to_device(self, model, device):
        return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt-oss-20b-lora")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=0, help="0이면 전체 사용")
    return parser.parse_args()


def load_training_samples(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError(f"학습 파일이 비어 있습니다: {path}")

    stripped = raw.lstrip()
    samples: List[Dict[str, Any]] = []

    if stripped.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON 배열 형식이어야 합니다.")
        samples = data
    else:
        for idx, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                samples.append(item)
            except Exception as e:
                raise ValueError(
                    f"{idx}번째 줄 JSON 파싱 실패: {e}\n문제 줄: {line[:300]}"
                ) from e

    validated = []
    for i, item in enumerate(samples):
        if not isinstance(item, dict):
            print(f"[경고] {i}번 샘플 제외: dict 아님")
            continue

        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"[경고] {i}번 샘플 제외: messages 없음")
            continue

        ok = True
        clean_messages = []
        for j, msg in enumerate(messages):
            if not isinstance(msg, dict):
                ok = False
                print(f"[경고] {i}번 샘플 {j} message 제외: dict 아님")
                break

            role = msg.get("role")
            content = msg.get("content")

            if role not in {"system", "user", "assistant", "developer", "tool"}:
                ok = False
                print(f"[경고] {i}번 샘플 {j} role 오류: {role}")
                break

            if not isinstance(content, str):
                ok = False
                print(f"[경고] {i}번 샘플 {j} content 오류")
                break

            clean_messages.append({"role": role, "content": content})

        if ok:
            validated.append({"messages": clean_messages})

    if not validated:
        raise ValueError("유효한 샘플이 없습니다.")

    return validated


def build_dataset(samples: List[Dict[str, Any]], tokenizer, max_length: int, max_samples: int = 0) -> Dataset:
    if max_samples > 0:
        samples = samples[:max_samples]

    rows = []
    skipped_too_long = 0

    for idx, ex in enumerate(samples):
        try:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            if len(token_ids) > max_length:
                skipped_too_long += 1
                continue

            rows.append({"text": text})
        except Exception as e:
            print(f"[경고] {idx}번 샘플 템플릿 처리 실패: {e}")

    print(f"전체 샘플 수: {len(samples)}")
    print(f"길이 초과 제외: {skipped_too_long}")
    print(f"최종 학습 샘플 수: {len(rows)}")

    if not rows:
        raise ValueError("최종 학습 샘플이 없습니다. max_length를 늘리거나 데이터 형식을 확인하세요.")

    return Dataset.from_list(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== tokenizer 로드 ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== model 로드 ===")
    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        use_cache=False,
        offload_buffers=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    # Trainer가 model parallel 로 인식하도록 힌트
    model.is_parallelizable = True
    model.model_parallel = True

    print("=== LoRA 적용 ===")
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "v_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("=== 학습 데이터 로드 ===")
    samples = load_training_samples(args.train_file)
    train_ds = build_dataset(
        samples=samples,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )

    print("=== trainer 설정 ===")
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = NoMoveSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
    )

    print("=== 학습 시작 ===")
    trainer.train()

    print("=== 저장 ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"완료: {args.output_dir}")


if __name__ == "__main__":
    main()
