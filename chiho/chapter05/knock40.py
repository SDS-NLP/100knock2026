"""
事前準備:
    python3 -m pip install google-genai
    export GEMINI_API_KEY="your-api-key"
"""

import os

from google import genai
#from google.genai import errors, types

# Zero-shot推論
prompt = """
9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア 藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ 嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ 藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

client = genai.Client()

response = client.models.generate_content_stream(
    model = "gemini-3.5-flash",
    contents = prompt
)

for chunk in response:
    print(chunk.text, end="", flush=True)
