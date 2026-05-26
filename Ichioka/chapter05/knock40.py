import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")

prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。
ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

def gen_answer(question: str) -> str:
    if API_KEY is None:
        raise ValueError("LLM_API_KEY not found!")

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