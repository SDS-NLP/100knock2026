import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

prompt = """
日本の近代化に関連するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　府知事・県令からなる地方官会議が設置された。
イ　廃藩置県が実施され，中央から府知事・県令が派遣される体制になった。
ウ　すべての藩主が，天皇に領地と領民を返還した。

解答: ウ→イ→ア
"""

def gen_answer(question: str) -> str:
    if API_KEY is None:
        raise ValueError("OPENROUTER_API_KEY not found!")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-OpenRouter-Title": "history-test",
        },
        json={
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
        },
        timeout=480,
    )

    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


if __name__ == "__main__":
    answer = gen_answer(prompt)
    print(answer)