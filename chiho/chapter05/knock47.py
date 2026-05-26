"""
事前準備
python3 -m pip install google-genai
export GEMINI_API_KEY="your-api-key"
"""

import os

from google import genai
#from google.genai import errors, types

MODEL_NAME = "gemini-2.5-flash"
client = genai.Client()

SENRYU_LIST = [
    "五月病 連休明けの 重い足",
    "大空に 夢を泳がす 鯉のぼり",
    "母の日や 普段言えない ありがとう",
    "風薫る 青葉まぶしい 五月晴れ",
    "柏餅 粒あん派だと 主張する",
    "連休は 過ぎてしまえば 夢の跡",
    "汗ばむ日 そろそろ準備 衣替え",
    "誇らしげ 居間に飾った 兜かな",
    "心地よい 薫風抜ける 散歩道",
    "連休で 財布の中身 五月晴れ",
]

for i, senryu in enumerate(SENRYU_LIST, start=1):
    prompt = f"""
あなたは川柳の評価者です。
次の川柳の面白さを10段階で評価してください。

出力形式:
score: <1から10の整数>
reason: <簡単な理由>

川柳:
{senryu}
""".strip()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    print(f"{i}. {senryu}")
    print(response.text)
    print()
