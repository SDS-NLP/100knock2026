from google import genai
# from google.genai import types
# from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

chat = client.chats.create(model="gemini-2.5-flash")

prompt_q44 ="""つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、間違えて急行に乗車してしまったことに気付きました。自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。目的地の駅の名前を答えてください。"""
response1 = chat.send_message(prompt_q44)

print(response1.text)

prompt_q45 = """さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"""
response2 = chat.send_message(prompt_q45)

print(response2.text)

"""つばめちゃんの目的地は **緑が丘 (Midorigaoka)** です。

理由：

1.  自由が丘から東急大井町線の大井町方面の急行に乗車した場合、次の停車駅は **大岡山 (Ookayama)** です。
2.  大岡山で降りて、反対方向（自由が丘方面）の電車で一駅戻った駅は **緑が丘 (Midorigaoka)** です。
この場合のつばめちゃんのルートを追ってみましょう。

1.  **自由が丘から間違えて乗車した急行電車（溝の口方面行き）の次の停車駅**
    東急大井町線の自由が丘駅から溝の口方面へ向かう急行は、次の停車駅は **二子玉川 (Futako-Tamagawa)** です。
    つばめちゃんは二子玉川で降車します。

2.  **二子玉川から目的地（緑が丘）への移動**
    目的地である緑が丘は、二子玉川から見ると自由が丘をさらに超えた大井町方面にあります。したがって、二子玉川から **大井町方面行きの各駅停車** に乗車します。

3.  **停車駅のカウント**
    二子玉川から大井町方面の各駅停車の駅を数えます。
    *   二子玉川（乗車駅）
    *   上野毛 (Kaminoge) - 1駅
    *   等々力 (Todoroki) - 2駅
    *   尾山台 (Oyamadai) - 3駅
    *   九品仏 (Kuho-butsu) - 4駅
    *   自由が丘 (Jiyugaoka) - 5駅
    *   **緑が丘 (Midorigaoka)** - 6駅

したがって、二子玉川から**6駅先**の駅で降りれば良いことになります。"""