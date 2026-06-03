import os
from google import genai
import statistics
import re
import time


def extract_scores(text):
    scores = []

    lines = text.split("\n")

    for line in lines:
        match = re.search(r"\)(\d+)", line)

        if match:
            scores.append(int(match.group(1)))

    return scores


def get_reply(prompt, output):
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    all_scores = []

    with open(output, "w") as f:
        f.write("results\n\n")

    for i in range(5):
        chat = client.chats.create(model="gemini-2.5-flash-lite")

        response = chat.send_message(prompt)

        print(f"--- {i+1}回目 ---")
        print(response.text)

        scores = extract_scores(response.text)
        all_scores.append(scores)

        with open(output, "a") as f:
            f.write(f"--- {i+1}回目 ---\n")
            f.write(response.text)
            f.write("\n\n")

        time.sleep(10)

    return all_scores


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

    output = "chapter05/output08.txt"

    results = get_reply(prompt, output)

    print("\n=== 分散 ===")

    for i in range(10):
        poem_scores = [result[i] for result in results]

        variance = statistics.variance(poem_scores)

        print(f"川柳{i+1}: {poem_scores} -> 分散={variance}")
