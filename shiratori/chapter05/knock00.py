import os
from google import genai


def get_reply(prompt, output):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    with open(output, "w") as f:
        f.write("results\n\n")

    response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)

    print(response.text)

    with open(output, "a") as f:
        f.write(response.text)


if __name__ == "__main__":
    prompt = """
        9世紀に活躍した人物に関係するできごとについて述べた次のア～ウを年代の古い順に正しく並べよ。

        ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。
        イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。
        ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。
        """
    output = "chapter05/output00.txt"
    get_reply(prompt, output)


# 9世紀に活躍した人物に関係するできごとを年代の古い順に並べると、以下のようになります。

# 1. **イ　嵯峨天皇は，藤原冬嗣らを蔵人頭に任命した。**
#    * 嵯峨天皇の在位は809年～823年です。蔵人頭の設置は810年頃であり、藤原冬嗣がその初代として任命されました。

# 2. **ウ　藤原良房は，承和の変後，藤原氏の中での北家の優位を確立した。**
#    * 承和の変は842年に起こりました。藤原良房はこの変で中心的な役割を果たし、藤原氏、特に北家の権力基盤を不動のものとしました。

# 3. **ア　藤原時平は，策謀を用いて菅原道真を政界から追放した。**
#    * 昌泰の変（菅原道真が左遷された事件）は901年に起こりました。藤原時平はこの事件の首謀者とされています。

# したがって、正しい順序は **イ → ウ → ア** です。
