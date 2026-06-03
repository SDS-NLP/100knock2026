import os
from dotenv import load_dotenv
from groq import Groq

MODEL = "qwen/qwen3-32b"
TEMPERATURE = 0.7
MAX_COMPLETION_TOKENS = 512

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")


SENRYU_LIST = [
    {
        "id": 1,
        "theme": "満員電車",
        "senryu": "満員の電車\n足が床にへばりつく\n朝の息吹",
    },
    {
        "id": 2,
        "theme": "寝坊",
        "senryu": "目覚まし無視し\n眠りの沼から這い出ず\n遅刻の汗つめたい",
    },
    {
        "id": 3,
        "theme": "大学生活",
        "senryu": "図書館の 電子ブックより 人恋しい",
    },
    {
        "id": 4,
        "theme": "アルバイト",
        "senryu": "コインランドリーで\nシフト入るたびに\nシワになる制服",
    },
    {
        "id": 5,
        "theme": "スマホ",
        "senryu": "電車の中で\nスマホに夢中なのは\n静かな群衆",
    },
    {
        "id": 6,
        "theme": "雨の日",
        "senryu": "雨足の音に 听き惚れす 静かな夕暮れ",
    },
    {
        "id": 7,
        "theme": "テスト前",
        "senryu": "テスト前　鉛筆を　三本用意",
    },
    {
        "id": 8,
        "theme": "コンビニ",
        "senryu": "深夜のコンビニ\n灯りが誘う空しさに\n缶が笑う",
    },
    {
        "id": 9,
        "theme": "夜更かし",
        "senryu": "深夜の鼠の音に 眠り難く 誰にも知らず",
    },
    {
        "id": 10,
        "theme": "春の朝",
        "senryu": "鶴の羽根吹く\n朝の光に桃の枝\n夢を染めて",
    },
]


def create_judge_prompt(item: dict) -> str:
    """川柳評価用のプロンプトを作る"""
    return f"""
    以下の川柳について、「面白さ」を10段階で評価してください。

    評価対象:
    お題：{item["theme"]}
    川柳：
    {item["senryu"]}

    出力形式:
    点数：X/10
    理由：...
    """


def judge_senryu(client: Groq, item: dict) -> str:
    """LLMに川柳の面白さを評価させる"""
    prompt = create_judge_prompt(item)

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "あなたは川柳コンテストの審査員です。"
                    "川柳の面白さを10段階で評価し、理由も簡潔に述べてください。"
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
        reasoning_effort="none",
    )

    return completion.choices[0].message.content.strip()


def main():
    groq = Groq(api_key=api_key)

    for item in SENRYU_LIST:
        result = judge_senryu(groq, item)

        print(f"{item['id']}. お題：{item['theme']}")
        print(item["senryu"])
        print()
        print(result)
        print("-" * 50)


if __name__ == "__main__":
    main()