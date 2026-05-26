import os
import re
import time
from collections import deque

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

CSV_PATH = "data/computer_security.csv"

MODEL = "qwen/qwen3-32b"

MAX_REQUESTS_PER_60_SECONDS = 50
WINDOW_SECONDS = 60

# 各リクエストの間隔
SLEEP_SECONDS_PER_REQUEST = 1

# 出力制御
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 16
RESULTS_CSV_PATH = "data/q43_jmmlu_groq_results.csv"

# APIエラー時のリトライ
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 10


def load_jmmlu_csv(csv_path: str) -> pd.DataFrame:
    """JMMLU/MMLU形式のCSVをヘッダーなしで読む。"""
    df = pd.read_csv(csv_path, header=None)

    df.columns = ["Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer"]

    # 念のため、文字列化・前後空白除去
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # 正解列を大文字化
    df["Answer"] = df["Answer"].str.upper()

    # A/B/C/D以外の正解があれば検出
    invalid_answers = df[~df["Answer"].isin(["A", "B", "C", "D"])]
    if len(invalid_answers) > 0:
        raise ValueError(
            "Answer列にA/B/C/D以外が含まれています。"
            f"該当行数: {len(invalid_answers)}"
        )

    return df


def build_prompt(row: pd.Series) -> str:
    """CSVの1行をAPI用プロンプトに変換する。"""
    return f"""次の多肢選択問題に答えてください。
    問題:
    {row["Question"]}

    選択肢:
    A. {row["Choice A"]}
    B. {row["Choice B"]}
    C. {row["Choice C"]}
    D. {row["Choice D"]}

    回答は A, B, C, D のうち1文字だけで出力してください。
    説明や句読点は不要です。"""


def parse_choice(text: str) -> str | None:
    """LLMの出力から A/B/C/D を抽出する。"""
    if text is None:
        return None

    text = text.strip()

    # まず、出力全体が1文字に近い場合を優先
    if re.fullmatch(r"[AaBbCcDd]", text):
        return text.upper()

    # 次に、単独の選択肢文字を抽出
    match = re.search(r"\b([AaBbCcDd])\b", text)
    if match:
        return match.group(1).upper()

    # 日本語や記号混じり: 「答え: C」「(D)」など
    match = re.search(r"[\(\[【「『\s:：]*([AaBbCcDd])[\)\]】」』\s。．,，]*", text)
    if match:
        return match.group(1).upper()

    return None


class RateLimiter:
    """簡易レートリミッター。"""

    def __init__(self, max_requests: int, window_seconds: int, sleep_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.sleep_seconds = sleep_seconds
        self.request_timestamps = deque()

    def wait_before_request(self):
        now = time.time()

        # window_secondsより古い記録を削除
        while self.request_timestamps and now - self.request_timestamps[0] >= self.window_seconds:
            self.request_timestamps.popleft()

        self.request_timestamps.append(time.time())

    def sleep_after_request(self):
        time.sleep(self.sleep_seconds)


def generate_answer(client: Groq, prompt: str) -> str:
    """Groq APIで回答を生成する!"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは多肢選択問題に答えるアシスタントです。"
                    "必ず A, B, C, D のうち1文字だけを出力してください。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        stream=False,
        reasoning_format="hidden",
        reasoning_effort="none"
    )

    return completion.choices[0].message.content


def generate_answer_with_retry(client: Groq, prompt: str) -> str:
    """リトライ"""
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return generate_answer(client, prompt)
        except Exception as e:
            last_error = e
            print(f"[error] attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(f"API呼び出しに失敗しました: {last_error}")


def main():
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "環境変数 GROQ_API_KEY が見つかりません。"
        )

    client = Groq(api_key=api_key)

    df = load_jmmlu_csv(CSV_PATH)

    print(f"問題数: {len(df)}")
    print(f"使用モデル: {MODEL}")

    rate_limiter = RateLimiter(
        max_requests=MAX_REQUESTS_PER_60_SECONDS,
        window_seconds=WINDOW_SECONDS,
        sleep_seconds=SLEEP_SECONDS_PER_REQUEST,
    )

    results = []
    correct_count = 0

    for idx, row in df.iterrows():
        question_no = idx + 1
        prompt = build_prompt(row)
        gold = row["Answer"]

        rate_limiter.wait_before_request()

        raw_output = generate_answer_with_retry(client, prompt)
        pred = parse_choice(raw_output)

        is_correct = pred == gold
        if is_correct:
            correct_count += 1

        accuracy_so_far = correct_count / question_no

        print(
            f"[{question_no}/{len(df)}] "
            f"pred={pred} gold={gold} "
            f"correct={is_correct} "
            f"acc={accuracy_so_far:.4f} "
            f"raw={raw_output!r}"
        )

        results.append(
            {
                "index": idx,
                "question": row["Question"],
                "choice_a": row["Choice A"],
                "choice_b": row["Choice B"],
                "choice_c": row["Choice C"],
                "choice_d": row["Choice D"],
                "gold": gold,
                "raw_output": raw_output,
                "pred": pred,
                "is_correct": is_correct,
            }
        )

        rate_limiter.sleep_after_request()

    final_accuracy = correct_count / len(df)

    print("-" * 60)
    print(f"正解数: {correct_count}/{len(df)}")
    print(f"正解率(%): {final_accuracy * 100:.2f}%")
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"結果CSVを保存したぜ: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()