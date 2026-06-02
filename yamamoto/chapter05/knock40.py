#以下の問題の解答を作成せよ。ただし、解答生成はzero-shot推論とせよ。

from google import genai
import gemini_api

prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""


client = genai.Client(api_key = gemini_api.api_key)

responce = client.models.generate_content(
    model = "gemini-3.1-flash-lite", contents = prompt
)

print("回答：")
print(responce.text)