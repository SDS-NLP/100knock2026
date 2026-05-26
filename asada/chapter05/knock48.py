import os
import statistics
from typing import List

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field


class EvaluationResult(BaseModel):
    score: List[int] = Field(description="番号順の川柳の評点(10点満点)")


load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

contents = """
以下の川柳に対する評価を出力してください.

1. AIが 言葉の裏を 読み切れず
2. 形態素 分かち書きして 意味探る
3. 翻訳の 直訳すぎて 吹き出す夜
4. 日本語は ディープなモデル 迷わせる
5. 主語抜けて 文脈だけが 頼り綱
6. 学習の データに混ざる バグと嘘
7. 同音の 異義語に惑う 機械かな
8. プロンプト 言葉遣いで 機嫌変え
9. 埋め込みの ベクトル空間 旅をする
10. 愛してる 確率いくつ？ 問い返す
"""

config = {
    "response_mime_type": "application/json",
    "response_schema": EvaluationResult,
    "temperature": 0.6,
}

tests = []
for i in range(5):
    print(f"In progress ... ({i + 1}/5)")
    response = client.models.generate_content(
        model="gemini-3.5-flash", contents=contents, config=config
    )
    tests.append(response.parsed)

for j in range(10):
    print(
        f"{j + 1}番目の川柳のスコアの分散: {statistics.variance(test.score[j] for test in tests)}"
    )

# output
# 1番目の川柳のスコアの分散: 0
# 2番目の川柳のスコアの分散: 0
# 3番目の川柳のスコアの分散: 0.2
# 4番目の川柳のスコアの分散: 0.2
# 5番目の川柳のスコアの分散: 0
# 6番目の川柳のスコアの分散: 0
# 7番目の川柳のスコアの分散: 0
# 8番目の川柳のスコアの分散: 0
# 9番目の川柳のスコアの分散: 0
# 10番目の川柳のスコアの分散: 0
