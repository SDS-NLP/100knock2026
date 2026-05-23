import os
from dotenv import load_dotenv
from groq import Groq

MODEL = "qwen/qwen3-32b"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 128

PROMPT = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

def generate_answer(client: Groq, prompt: str) -> str:
    """Groq APIで回答を生成する!"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                   "この問いかけに回答してください。答えのみを出力してください。"
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

def main(prompt: str):
    groq = Groq(api_key=api_key)
    answer = generate_answer(groq, prompt)
    print(answer)

if __name__ == "__main__":
    main(PROMPT)