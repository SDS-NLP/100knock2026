# 対話

prompt = '''
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。
目的地の駅の名前を答えてください。
'''

import os

from google import genai
#from google.genai import errors, types

# Zero-shot推論

client = genai.Client()

response = client.models.generate_content_stream(
    model = "gemini-3.5-flash",
    contents = prompt
)

for chunk in response:
    print(chunk.text, end="", flush=True)