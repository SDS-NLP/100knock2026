import os
from google import genai

client = genai.Client()

prompt = """
以下の日本史の問題に対して、正解と、その結論に至る各出来事の年代（西暦）および簡単な解説を出力してください。
思考プロセスも含めて段階的に説明してください。

【問題】
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア 藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ 嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ 藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=prompt,
)

print(response.text)