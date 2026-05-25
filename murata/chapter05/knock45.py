import os 
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

def generate(prompt, temperature=None):
    cfg = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    res = _client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    return res.text
q1 = """つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"""

q2 = """さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"""

chat = _client.chats.create(model=MODEL)
print("--- 1ターン目 ---")
print(chat.send_message(q1).text)
print("\n--- 2ターン目 ---")
print(chat.send_message(q2).text)