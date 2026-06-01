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
    以下の川柳の面白さを10段階で評価せよ。返答は次のように答えよ→川柳1)5
    ## 初夏をテーマにした川柳10選
    # 1. **薫風に 心も軽し 散歩道**
   2. **青田風 稲穂のさざめき 誘い水**
   3. **若葉萌え 木漏れ日浴びて 昼寝かな**
   4. **五月晴れ 遠くに見える 山笑う**
   5. **雨上がり 虹を眺めて 傘しまう**
   6. **蛍火や 闇夜に光る 儚さよ**
   7. **庭先で 蝉の声聞く 夏近し**
   8. **緑濃く 鮮やかなるは 山の色**
   9. **汗ばんで 飲む麦茶は 五臓に染む**
   10. **初夏来たる 昼の長さに 喜びぬ**
    """
    prompt = prompt.replace("*", "")
    print(prompt)
    output = "chapter05/output07.txt"
    get_reply(prompt, output)
