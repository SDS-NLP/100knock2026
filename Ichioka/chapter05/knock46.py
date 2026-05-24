import os
from dotenv import load_dotenv
from groq import Groq

MODEL = "qwen/qwen3-32b"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 128

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


def create_prompt(theme: str) -> str:
    """テーマから川柳生成用のプロンプトを作る"""
    return f"""
    お題は「{theme}」です。

    このお題に沿って、川柳を1つ作成してください。
    川柳は五・七・五を意識してください。
    出力は川柳のみとし、説明は不要です。
    """


def generate_answer(client: Groq, prompt: str) -> str:
    """Groq APIで回答を生成する"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "あなたは川柳を作るのが得意な作家です。出力は川柳のみとしてください。",
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
        reasoning_effort="none",
    )

    return completion.choices[0].message.content.strip()


def main():
    groq = Groq(api_key=api_key)

    themes = [
        "満員電車",
        "寝坊",
        "大学生活",
        "アルバイト",
        "スマホ",
        "雨の日",
        "テスト前",
        "コンビニ",
        "夜更かし",
        "春の朝",
    ]

    for i, theme in enumerate(themes, start=1):
        prompt = create_prompt(theme)
        senryu = generate_answer(groq, prompt)

        print(f"{i}. お題：{theme}")
        print(senryu)
        print("-" * 30)


if __name__ == "__main__":
    main()