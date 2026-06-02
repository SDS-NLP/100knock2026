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
prompt_1 = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
つばめちゃんの目的地でした。目的地の駅の名前を答えてください。

以下のような形式のみで回答してください。
1. 駅の名前
"""

prompt_2 = """
さらに、つばめちゃんが自由が丘駅で乗り換えたとき、
先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、
反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？

以下のような形式のみで回答してください。
1. その駅までの駅数
"""

response_1 = model.generate_content(prompt_1)
response_2 = model.generate_content(prompt_2)
print(response_1.text)
print(response_2.text)