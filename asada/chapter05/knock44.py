import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

contents = """
つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。
"""
response = client.models.generate_content_stream(
    model="gemini-3.5-flash", contents=contents
)

for chunk in response:
    print(chunk.text, end="", flush=True)

# output
# つばめちゃんの目的地の駅は、**緑が丘（みどりがおか）駅**です。
#
# **【解説】**
# 1. **自由が丘駅**から大井町方面（東行）の急行に乗車します。
# 2. 自由が丘の次の急行停車駅は、東急目黒線との接続駅でもある**大岡山（おおおかやま）駅**です。
# 3. 大岡山駅で降り、反対方向（二子玉川・溝の口方面）の各駅停車に乗り換えて一駅戻ると、**緑が丘駅**に到着します。
