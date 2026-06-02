import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Gemini APIの設定
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# 問題文
prompt = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
つばめちゃんの目的地でした。目的地の駅の名前を答えてください。

以下のような形式で回答してください。
1. 駅の名前
"""

response = model.generate_content(prompt)
print(response.text)