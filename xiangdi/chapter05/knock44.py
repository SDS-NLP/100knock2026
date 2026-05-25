import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """
以下の問いかけに対する応答を生成せよ。

つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、
各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅が
つばめちゃんの目的地でした。

目的地の駅の名前を答えてください。
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

print(response.text)