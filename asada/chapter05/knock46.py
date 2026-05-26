import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

contents = """
自然言語処理をお題として、川柳のみ番号付きで１０個出力せよ。
"""
response = client.models.generate_content_stream(
    model="gemini-3.5-flash", contents=contents
)

for chunk in response:
    print(chunk.text, end="", flush=True)

# output
# 1. AIが 言葉の裏を 読み切れず
#
# 2. 形態素 分かち書きして 意味探る
#
# 3. 翻訳の 直訳すぎて 吹き出す夜
#
# 4. 日本語は ディープなモデル 迷わせる
#
# 5. 主語抜けて 文脈だけが 頼り綱
#
# 6. 学習の データに混ざる バグと嘘
#
# 7. 同音の 異義語に惑う 機械かな
#
# 8. プロンプト 言葉遣いで 機嫌変え
#
# 9. 埋め込みの ベクトル空間 旅をする
#
# 10. 愛してる 確率いくつ？ 問い返す
