from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# prompt="""以下のお題に沿って１０個の川柳を作成せよ。
#         お題：大学生
#         川柳："""

# response = client.models.generate_content(
#     model="gemini-2.5-flash",
#     contents=prompt
# )

# print(response.text)

theme = "大学生"
prompt_46 = f"お題「{theme}」で、5・7・5の川柳をちょうど10個作成してください。"

print(f"【問題46】お題「{theme}」で川柳を生成中...\n")
response_46 = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt_46,
    config=types.GenerateContentConfig(
        temperature=0.7 
    )
)
"""はい、承知いたしました。「大学生」をテーマにした川柳を10個作成します。

---

1.  朝授業　行けば単位は　来るけれど

2.  卒業が　迫る単位と　就活と

3.  バイトして　稼いだ金は　すぐ消える

4.  飲み会の　次の日いつも　記憶なし

5.  目覚ましも　聞こえず昼が　夜になる

6.  レポートは　締切前に　覚醒す

7.  出席点　それだけ頼りに　行く授業

8.  仕送りで　生き延びる日々　親に感謝

9.  将来を　語る友見て　焦る夜

10. 夏休み　気づけば終わる　何もせず"""