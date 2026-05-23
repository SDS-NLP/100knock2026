import os
from dotenv import load_dotenv
from groq import Groq

MODEL = "qwen/qwen3-32b"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 128

PROMPT = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""

FOLLOW_UP_PROMPT = """
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？
"""

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


def generate_answer(client: Groq, prompt: str) -> str:
    """Groq APIで最初の回答を生成する"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "この問いかけに回答してください。答えのみを出力してください。",
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


def generate_follow_up_answer(
    client: Groq,
    original_prompt: str,
    first_answer: str,
    follow_up_prompt: str,
) -> str:
    """最初の応答に続けて、追加の問いかけへの回答を生成する"""
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "この問いかけに回答してください。答えのみを出力してください。",
            },
            {
                "role": "user",
                "content": original_prompt,
            },
            {
                "role": "assistant",
                "content": first_answer,
            },
            {
                "role": "user",
                "content": follow_up_prompt,
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

    first_answer = generate_answer(groq, PROMPT)
    print("1回目の回答:")
    print(first_answer)

    follow_up_answer = generate_follow_up_answer(
        client=groq,
        original_prompt=PROMPT,
        first_answer=first_answer,
        follow_up_prompt=FOLLOW_UP_PROMPT,
    )

    print("\n追加質問への回答:")
    print(follow_up_answer)


if __name__ == "__main__":
    main()