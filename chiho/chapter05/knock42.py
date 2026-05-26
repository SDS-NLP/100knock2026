"""
JMMLUの1科目をGemini APIで解かせ、正解率を求める

事前準備:
    python3 -m pip install google-genai certifi
    export GEMINI_API_KEY="your-api-key"

実行例:
    python3 chiho/chapter05/knock42.py
    python3 chiho/chapter05/knock42.py japanese_history 20
"""

#まだおかしいので修正する

from __future__ import annotations

import csv
import io
import os
import re
import ssl
import sys
import time
import urllib.request
from enum import Enum

import certifi
from google import genai
from google.genai import errors, types


MODEL_NAME = "gemini-2.5-flash"
DEFAULT_SUBJECT = "japanese_history"
DEFAULT_LIMIT = 5
BASE_URL = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU"
CHOICES = ("A", "B", "C", "D")
MAX_RETRIES = 3


class ChoiceEnum(Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


def load_dataset(subject: str) -> list[dict[str, str]]:
    url = f"{BASE_URL}/{subject}.csv"
    ssl_context = ssl.create_default_context(cafile=certifi.where())

    with urllib.request.urlopen(url, context=ssl_context) as response:
        text = response.read().decode("utf-8-sig")

    reader = csv.reader(io.StringIO(text))
    rows = []

    for row in reader:
        if len(row) < 6:
            continue
        rows.append(
            {
                "question": row[0].strip(),
                "A": row[1].strip(),
                "B": row[2].strip(),
                "C": row[3].strip(),
                "D": row[4].strip(),
                "answer": row[5].strip().upper(),
            }
        )

    if not rows:
        raise ValueError(f"科目 {subject} のデータを読み込めませんでした。")

    if any(row["answer"] not in CHOICES for row in rows):
        raise ValueError("正解ラベルが A, B, C, D 以外の行を検出しました。")

    return rows


def build_prompt(example: dict[str, str]) -> str:
    return f"""
次の4択問題に答えてください。
出力は A, B, C, D のいずれか1文字だけにしてください。
説明は不要です。

問題:
{example["question"]}

A. {example["A"]}
B. {example["B"]}
C. {example["C"]}
D. {example["D"]}
""".strip()


def extract_choice(text: str) -> str | None:
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def solve_question(client: genai.Client, example: dict[str, str]) -> str | None:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=build_prompt(example),
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1,
            response_mime_type="text/x.enum",
            response_schema=ChoiceEnum,
        ),
    )

    if not response.text:
        return None

    return extract_choice(response.text)


def get_retry_wait_seconds(message: str) -> float:
    match = re.search(r"Please retry in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 1.0
    return 60.0


def main() -> None:
    subject = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SUBJECT
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_LIMIT
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY もしくは GOOGLE_API_KEY を環境変数に設定してください。"
        )

    try:
        dataset = load_dataset(subject)[:limit]
    except Exception as error:
        raise SystemExit(f"JMMLUデータセットの読み込みに失敗しました: {error}") from error

    correct = 0
    answered = 0

    try:
        with genai.Client(api_key=api_key) as client:
            for index, example in enumerate(dataset, start=1):
                prediction = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        prediction = solve_question(client, example)
                        break
                    except errors.APIError as error:
                        if error.code == 429 and attempt < MAX_RETRIES:
                            wait_seconds = get_retry_wait_seconds(error.message)
                            print(
                                f"{index:03d} rate_limited: "
                                f"{wait_seconds:.1f}s 待って再試行します"
                            )
                            time.sleep(wait_seconds)
                            continue
                        raise

                gold = example["answer"]

                is_correct = prediction == gold
                if prediction is not None:
                    answered += 1
                if is_correct:
                    correct += 1

                print(
                    f"{index:03d} pred={prediction or '-'} gold={gold} "
                    f"{'OK' if is_correct else 'NG'}"
                )
    except errors.APIError as error:
        raise SystemExit(f"Gemini API error ({error.code}): {error.message}") from error

    total = len(dataset)
    accuracy = correct / total if total else 0.0

    print("\n=== result ===")
    print(f"subject: {subject}")
    print(f"model: {MODEL_NAME}")
    print(f"total: {total}")
    print(f"answered: {answered}")
    print(f"correct: {correct}")
    print(f"accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
