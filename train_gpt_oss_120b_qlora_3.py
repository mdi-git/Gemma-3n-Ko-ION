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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--train_file", type=str, default="train.jsonl")
    parser.add_argument("--output_dir", type=str, default="./outputs/gpt-oss-120b-lora")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=0, help="0이면 전체 사용")
    return parser.parse_args()


def load_training_samples(path: str) -> List[Dict[str, Any]]:
    """
    다음 두 형식을 모두 허용:
    1) JSONL:
       {"messages":[...]}
       {"messages":[...]}
    2) JSON array:
       [
         {"messages":[...]},
         {"messages":[...]}
       ]
    빈 줄, UTF-8 BOM도 최대한 안전하게 처리.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"학습 파일을 찾을 수 없습니다: {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        raw = f.read()

    if not raw.strip():
        raise ValueError(f"학습 파일이 비어 있습니다: {path}")

    # 먼저 전체 파일이 JSON array/object인지 시도
    stripped = raw.lstrip()
    samples: List[Dict[str, Any]] = []

    if stripped.startswith("["):
        try:
            data = json.loads(raw)
            if not isinstance(data, list):
                raise ValueError("JSON 배열 형식이어야 합니다.")
            samples = data
        except Exception as e:
            raise ValueError(f"JSON 배열 파싱 실패: {e}") from e
    else:
        # JSONL로 처리
        lines = raw.splitlines()
        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                samples.append(item)
            except Exception as e:
                raise ValueError(
                    f"{idx}번째 줄 JSON 파싱 실패: {e}\n"
                    f"문제 줄 내용: {line[:300]}"
                ) from e

    if not samples:
        raise ValueError("유효한 학습 샘플이 하나도 없습니다.")

    # 형식 검증
    validated = []
    for i, item in enumerate(samples):
        if not isinstance(item, dict):
            print(f"[경고] {i}번 샘플은 dict가 아니어서 제외합니다.")
            continue

        messages = item.get("messages")
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"[경고] {i}번 샘플은 messages가 없거나 비어 있어 제외합니다.")
            continue

        ok = True
        clean_messages = []
        for j, msg in enumerate(messages):
            if not isinstance(msg, dict):
                ok = False
                print(f"[경고] {i}번 샘플의 {j}번 message가 dict가 아닙니다. 제외합니다.")
                break

            role = msg.get("role")
            content = msg.get("content")

            if role not in {"system", "user", "assistant", "developer", "tool"}:
                ok = False
                print(f"[경고] {i}번 샘플의 {j}번 message role이 잘못되었습니다: {role}")
                break

            if not isinstance(content, str):
                ok = False
                print(f"[경고] {i}번 샘플의 {j}번 message content가 문자열이 아닙니다.")
                break

            clean_messages.append({"role": role, "content": content})

        if ok:
            validated.append({"messages": clean_messages})

    if not validated:
        raise ValueError("형식 검증을 통과한 샘플이 없습니다.")

    return validated


def build_dataset(samples: List[Dict[str, Any]], tokenizer, max_length: int, max_samples: int = 0) -> Dataset:
    rows = []

    if max_samples > 0:
        samples = samples[:max_samples]

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
            print(f"[경고] {idx}번 샘플 템플릿 적용 실패: {e}")
            continue

    print(f"전체 샘플 수: {len(samples)}")
    print(f"길이 초과로 제외된 샘플 수: {skipped_too_long}")
    print(f"최종 학습 샘플 수: {len(rows)}")

    if not rows:
        raise ValueError(
            "최종 학습 샘플이 없습니다. "
            "train_file 형식이나 max_length 값을 확인하세요."
        )

    return Dataset.from_list(rows)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== tokenizer 로드 ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=== model 로드 ===")
    print("MXFP4 -> dequantize=True 로 로드합니다.")
    quantization_config = Mxfp4Config(dequantize=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto",
        use_cache=False,
        offload_buffers=True,
    )

    model.config.use_cache = False

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
        warmup_ratio=args.warmup_ratio,
        report_to="none",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
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
