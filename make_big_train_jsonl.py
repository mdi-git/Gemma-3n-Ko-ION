import os
import json
import random
import argparse
from typing import List, Dict

KOREAN_TOPICS = [
    "파이썬", "장고", "패스트API", "리눅스", "라즈베리파이", "MQTT", "Redis", "Celery",
    "벡터 검색", "임베딩", "트랜스포머", "자연어처리", "멀티모달", "데이터베이스",
    "네트워크", "OpenCV", "OCR", "전력 예측", "태양광", "풍력", "마이크로그리드",
    "시계열 분석", "조선시대 의례", "왕실 기록", "문헌 해석", "오디오 앰프", "진공관",
    "기타 이펙터", "스피커", "인공지능", "서버 운영", "도커", "엔진엑스"
]

SYSTEM_PROMPTS = [
    "너는 정확하고 친절한 한국어 기술 어시스턴트다.",
    "너는 간결하지만 핵심을 잘 설명하는 한국어 AI 조수다.",
    "너는 실무 중심으로 답하는 기술 문서 작성 도우미다.",
    "너는 학습용 예시를 풍부하게 만드는 한국어 어시스턴트다.",
    "너는 사용자의 질문에 단계적으로 답하는 전문가형 비서다."
]

USER_TEMPLATES = [
    "{topic}에 대해 설명해줘.",
    "{topic}의 기본 개념을 알려줘.",
    "{topic}를 실무에서 어떻게 활용하는지 알려줘.",
    "{topic}의 장점과 단점을 정리해줘.",
    "{topic}를 처음 배우는 사람에게 쉽게 설명해줘.",
    "{topic} 관련 예시를 들어서 설명해줘.",
    "{topic}를 사용할 때 주의할 점을 알려줘.",
    "{topic}와 관련된 핵심 용어를 정리해줘.",
    "{topic}를 적용하는 절차를 단계별로 설명해줘.",
    "{topic}에 대해 긴 설명문을 작성해줘."
]

SENTENCE_FRAGMENTS = [
    "이 개념은 실제 환경에서 자주 사용된다",
    "설계 단계에서는 요구사항을 먼저 정리해야 한다",
    "구현 단계에서는 구조를 단순하게 유지하는 편이 좋다",
    "운영 환경에서는 로그와 모니터링이 중요하다",
    "성능을 높이려면 병목 지점을 먼저 파악해야 한다",
    "보안을 위해 입력 검증과 접근 제어가 필요하다",
    "확장성을 고려하면 모듈화를 해두는 것이 유리하다",
    "문제가 생기면 재현 가능한 최소 사례를 만들어 확인해야 한다",
    "테스트 자동화는 장기 유지보수에 큰 도움이 된다",
    "구성 요소 사이의 책임 분리가 중요하다",
    "데이터 품질은 결과 성능에 직접적인 영향을 준다",
    "현실적인 제약 조건을 반영해 설계를 조정해야 한다",
    "캐시 전략은 응답 속도 개선에 유효할 수 있다",
    "병렬 처리와 비동기 처리를 구분해서 적용해야 한다",
    "작은 단위부터 검증하면 전체 시스템 안정성이 높아진다",
    "문서화를 해두면 협업 효율이 크게 향상된다",
    "설정값은 코드와 분리해 관리하는 것이 바람직하다",
    "문제를 해결할 때는 관찰 가능한 지표를 확보해야 한다",
    "장애 대응 절차를 미리 정리해두면 운영 부담이 줄어든다",
    "단순한 구조가 장기적으로 더 강한 경우가 많다"
]


def make_user_prompt() -> str:
    topic = random.choice(KOREAN_TOPICS)
    template = random.choice(USER_TEMPLATES)
    return template.format(topic=topic)


def make_assistant_answer(min_sentences: int, max_sentences: int) -> str:
    topic = random.choice(KOREAN_TOPICS)
    n = random.randint(min_sentences, max_sentences)

    parts = []
    intro_templates = [
        f"{topic}는 여러 환경에서 활용되는 주제다.",
        f"{topic}를 이해하려면 개념, 구조, 활용 사례를 함께 보는 것이 좋다.",
        f"{topic}는 이론과 실무가 모두 중요한 분야다.",
        f"{topic}를 설명할 때는 정의와 적용 맥락을 함께 다루는 편이 이해에 도움이 된다."
    ]
    parts.append(random.choice(intro_templates))

    for _ in range(n):
        frag = random.choice(SENTENCE_FRAGMENTS)
        suffix_templates = [
            f"특히 {topic}를 다룰 때 이 점을 놓치지 않는 것이 중요하다.",
            f"이러한 관점은 {topic}를 실제로 적용할 때 유용하다.",
            f"따라서 {topic}를 설계하거나 운용할 때 참고할 수 있다.",
            f"이 점은 {topic}의 성능과 안정성을 함께 고려할 때 의미가 있다.",
            f"결국 {topic}를 다룰 때는 기본 원리와 운영 측면을 함께 봐야 한다."
        ]
        parts.append(f"{frag}. {random.choice(suffix_templates)}")

    outro_templates = [
        f"정리하면 {topic}는 기초 개념, 구현 방법, 운영 전략을 함께 이해해야 효과적으로 활용할 수 있다.",
        f"결론적으로 {topic}는 단순한 이론이 아니라 실제 문제 해결에 직접 연결되는 주제다.",
        f"요약하면 {topic}는 구조적 이해와 반복적인 실습이 중요하다.",
        f"따라서 {topic}를 익힐 때는 작은 예제를 통해 점진적으로 확장하는 것이 좋다."
    ]
    parts.append(random.choice(outro_templates))

    return " ".join(parts)


def approximate_token_target_text(target_tokens: int) -> str:
    """
    대략적인 분량 확보용 텍스트 생성.
    한국어 기준 토큰 수를 정확히 맞추지는 않지만,
    긴 응답을 만들기 위한 보조 함수.
    """
    chunks = []
    while len(" ".join(chunks)) < target_tokens * 3:
        chunks.append(make_assistant_answer(8, 18))
    return " ".join(chunks)


def build_sample(max_tokens_per_answer: int) -> Dict:
    system = random.choice(SYSTEM_PROMPTS)
    user = make_user_prompt()

    # 대략적인 응답 길이 조절
    if max_tokens_per_answer <= 128:
        assistant = make_assistant_answer(3, 6)
    elif max_tokens_per_answer <= 256:
        assistant = make_assistant_answer(6, 12)
    elif max_tokens_per_answer <= 512:
        assistant = make_assistant_answer(12, 24)
    else:
        assistant = approximate_token_target_text(max_tokens_per_answer)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def generate_jsonl(
    output_file: str,
    num_samples: int,
    max_tokens_per_answer: int,
    flush_every: int = 1000
):
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(1, num_samples + 1):
            sample = build_sample(max_tokens_per_answer)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

            if i % flush_every == 0:
                f.flush()
                print(f"[진행] {i:,}/{num_samples:,} 샘플 생성 완료")

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n완료: {output_file}")
    print(f"샘플 수: {num_samples:,}")
    print(f"파일 크기: {size_mb:,.2f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="train_big.jsonl")
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--max_tokens_per_answer", type=int, default=256)
    parser.add_argument("--flush_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    generate_jsonl(
        output_file=args.output,
        num_samples=args.num_samples,
        max_tokens_per_answer=args.max_tokens_per_answer,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
