import os
from google import genai

MODEL = 'gemini-3.1-flash-lite'
client = genai.Client(api_key=os.environ['API_KEY'])

question = """9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。

年代の古い順にア・イ・ウを並べよ。「X→Y→Z」の形式で答えよ。"""

response = client.models.generate_content(
    model=MODEL,
    contents=question,
    config={'temperature': 0},
)
print(response.text)

"""正解は **イ→ウ→ア** です。

それぞれの出来事の年代は以下の通りです。

*   **イ：810年（弘仁元年）**
    嵯峨天皇が薬子の変の際に、蔵人所を設置し、藤原冬嗣らを初代の蔵人頭に任命しました。
*   **ウ：842年（承和9年）**
    承和の変で伴健岑や橘逸勢らを排斥し、藤原良房が権力を掌握して北家の優位を確立しました。
*   **ア：901年（昌泰4年）**
    昌泰の変で、藤原時平の策謀により菅原道真が大宰府へ左遷されました。"""