from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


prompt = """
        9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

        ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
        イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
        ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)

"""年代順に並べると以下のようになります。

*   **イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。**
    *   これは薬子の変の直後、**810年**に蔵人所が設置された時の出来事です。

*   **ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。**
    *   承和の変は**842年**に起こりました。

*   **ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。**
    *   これは昌泰の変（菅原道真の左遷）で、**901年**の出来事です。

したがって、古い順に並べると以下の通りです。"""

# import google.generativeai as genai

# # APIキーの設定（ご自身のキーを入れてください）
# genai.configure(api_key="AIzaSyDscznzBVilr6Wj5zI0Wjqy8D8wov92LKk")

# print("--- 利用可能なモデル一覧 ---")
# for m in genai.list_models():
#     # generateContent（テキスト生成）に対応しているモデルのみ絞り込み
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

"""年代の古い順に並べると以下のようになります。

*   **イ**　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
    *   これは弘仁元年（**810年**）のことです。薬子の変の後に蔵人所が設置され、藤原冬嗣が初代蔵人頭に任命されました。

*   **ウ**　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
    *   承和の変は承和9年（**842年**）に起こりました。この変によって藤原良房は政治的地位を固め、藤原北家の勢力を確立しました。

*   **ア**　藤原時平は，策謀を用いて菅原道真を政界から追放した。
    *   菅原道真が大宰府に左遷されたのは延喜元年（**901年**）のことです。これは10世紀初頭の出来事ですが、藤原時平も菅原道真も9世紀末から活躍した人物です。

したがって、年代の古い順に並べると **イ → ウ → ア** となります。"""