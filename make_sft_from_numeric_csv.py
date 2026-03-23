import os
import json
import math
import argparse
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


def safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def summarize_series(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float32)
    if len(arr) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "first": 0.0,
            "last": 0.0,
            "delta": 0.0,
        }

    return {
        "count": int(len(arr)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "first": float(arr[0]),
        "last": float(arr[-1]),
        "delta": float(arr[-1] - arr[0]),
    }


def trend_label(first: float, last: float, std: float) -> str:
    delta = last - first
    threshold = max(std * 0.3, 1e-6)

    if delta > threshold:
        return "상승"
    elif delta < -threshold:
        return "하락"
    return "횡보"


def volatility_label(std: float, mean_abs: float) -> str:
    base = max(mean_abs, 1e-6)
    ratio = std / base
    if ratio < 0.03:
        return "낮음"
    elif ratio < 0.10:
        return "보통"
    return "높음"


def detect_simple_anomaly(values: List[float]) -> str:
    arr = np.array(values, dtype=np.float32)
    if len(arr) < 5:
        return "판단 보류"

    mean = arr.mean()
    std = arr.std()
    if std < 1e-6:
        return "이상 징후 낮음"

    z = np.abs((arr - mean) / std)
    if z.max() >= 3.5:
        return "이상 징후 있음"
    elif z.max() >= 2.5:
        return "주의 필요"
    return "이상 징후 낮음"


def build_prompt(window_df: pd.DataFrame, feature_cols: List[str], time_col: Optional[str]) -> str:
    lines = []
    if time_col and time_col in window_df.columns:
        start_t = str(window_df.iloc[0][time_col])
        end_t = str(window_df.iloc[-1][time_col])
        lines.append(f"관측 구간: {start_t} ~ {end_t}")

    for col in feature_cols:
        vals = [safe_float(v) for v in window_df[col].tolist()]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue

        # 너무 길면 다운샘플링
        if len(vals) > 64:
            idx = np.linspace(0, len(vals) - 1, 64).astype(int)
            vals = [vals[i] for i in idx]

        rounded = [round(v, 4) for v in vals]
        lines.append(f"{col}: {rounded}")

    lines.append("위 시계열 구간의 추세, 변동성, 이상 징후 가능성을 한국어로 요약해줘.")
    return "\n".join(lines)


def build_answer(window_df: pd.DataFrame, feature_cols: List[str]) -> str:
    sections = []

    for col in feature_cols:
        vals = [safe_float(v) for v in window_df[col].tolist()]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue

        s = summarize_series(vals)
        t = trend_label(s["first"], s["last"], s["std"])
        v = volatility_label(s["std"], abs(s["mean"]))
        a = detect_simple_anomaly(vals)

        section = (
            f"{col}는 전반적으로 {t} 추세를 보인다. "
            f"평균은 {s['mean']:.4f}, 최솟값은 {s['min']:.4f}, 최댓값은 {s['max']:.4f}이다. "
            f"표준편차는 {s['std']:.4f}로 변동성은 {v} 수준이다. "
            f"구간 시작값은 {s['first']:.4f}, 종료값은 {s['last']:.4f}이며 변화량은 {s['delta']:.4f}이다. "
            f"이상 징후 평가는 '{a}'로 볼 수 있다."
        )
        sections.append(section)

    if not sections:
        return "유효한 수치 데이터가 부족하여 분석할 수 없다."

    return " ".join(sections)


def process_csv_to_jsonl(
    input_csv: str,
    output_jsonl: str,
    feature_cols: List[str],
    time_col: Optional[str],
    window_size: int,
    stride: int,
    max_samples: int,
    chunksize: int,
):
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    written = 0
    total_rows = 0
    overlap_buffer = None

    with open(output_jsonl, "w", encoding="utf-8") as fout:
        for chunk in pd.read_csv(input_csv, chunksize=chunksize):
            total_rows += len(chunk)

            if overlap_buffer is not None:
                chunk = pd.concat([overlap_buffer, chunk], ignore_index=True)

            valid_cols = [c for c in feature_cols if c in chunk.columns]
            if not valid_cols:
                raise ValueError(f"지정한 feature_cols가 CSV에 없습니다: {feature_cols}")

            # 다음 청크와 이어붙이기 위한 버퍼
            if len(chunk) >= window_size:
                overlap_buffer = chunk.iloc[-(window_size - 1):].copy()
            else:
                overlap_buffer = chunk.copy()

            max_start = len(chunk) - window_size
            if max_start < 0:
                continue

            for start in range(0, max_start + 1, stride):
                window_df = chunk.iloc[start:start + window_size].copy()

                prompt = build_prompt(window_df, valid_cols, time_col)
                answer = build_answer(window_df, valid_cols)

                sample = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "너는 대규모 수치 시계열을 읽고 추세와 이상 징후를 설명하는 한국어 분석 어시스턴트다."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        },
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    ]
                }

                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written += 1

                if written % 1000 == 0:
                    fout.flush()
                    print(f"[진행] {written:,} samples written")

                if max_samples > 0 and written >= max_samples:
                    print(f"[완료] 최대 샘플 수 {max_samples:,}에 도달")
                    print(f"[참고] 읽은 원본 행 수: {total_rows:,}")
                    return

    print(f"[완료] 총 생성 샘플 수: {written:,}")
    print(f"[참고] 읽은 원본 행 수: {total_rows:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, default="numeric_sft.jsonl")
    parser.add_argument("--feature_cols", type=str, required=True,
                        help="쉼표로 구분된 수치 컬럼명 예: power,wind,temp")
    parser.add_argument("--time_col", type=str, default="")
    parser.add_argument("--window_size", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max_samples", type=int, default=0, help="0이면 제한 없음")
    parser.add_argument("--chunksize", type=int, default=200000)
    args = parser.parse_args()

    feature_cols = [x.strip() for x in args.feature_cols.split(",") if x.strip()]
    time_col = args.time_col.strip() or None

    process_csv_to_jsonl(
        input_csv=args.input_csv,
        output_jsonl=args.output_jsonl,
        feature_cols=feature_cols,
        time_col=time_col,
        window_size=args.window_size,
        stride=args.stride,
        max_samples=args.max_samples,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
