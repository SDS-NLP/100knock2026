import os
from google import genai
from google.genai import types

MODEL = 'gemini-3.1-flash-lite'
client = genai.Client(api_key=os.environ['API_KEY'])

turn1 = """つばめちゃんは渋谷駅から東急東横線に乗り、自由が丘駅で乗り換えました。
東急大井町線の大井町方面の電車に乗り換えたとき、各駅停車に乗車すべきところ、
間違えて急行に乗車してしまったことに気付きました。
自由が丘の次の急行停車駅で降車し、反対方向の電車で一駅戻った駅がつばめちゃんの目的地でした。
つばめちゃんの目的地の駅名は何ですか？"""

response1 = client.models.generate_content(
    model=MODEL,
    contents=turn1,
    config={'temperature': 0},
)
print('Turn 1:')
print(response1.text)

turn2 = """さらに、つばめちゃんが自由が丘駅で乗り換えたとき、先ほどとは反対方向の急行電車に間違って乗車してしまった場合を考えます。
目的地の駅に向かうため、自由が丘の次の急行停車駅で降車した後、反対方向の各駅停車に乗車した場合、何駅先の駅で降りれば良いでしょうか？"""

history = [
    types.Content(role='user', parts=[types.Part(text=turn1)]),
    types.Content(role='model', parts=[types.Part(text=response1.text)]),
    types.Content(role='user', parts=[types.Part(text=turn2)]),
]

response2 = client.models.generate_content(
    model=MODEL,
    contents=history,
    config={'temperature': 0},
)
print('\nTurn 2:')
print(response2.text)

"""Turn 1:
つばめちゃんの目的地の駅名は、**九品仏（くほんぶつ）駅**です。

理由は以下の通りです。

1.  自由が丘駅から大井町線の大井町方面へ向かう急行の停車駅は、**「大岡山」**です。
2.  つばめちゃんは自由が丘の次の急行停車駅である「大岡山」で降車しました。
3.  そこから反対方向（自由が丘方面）の電車で一駅戻った駅は、大岡山駅の隣の駅である「九品仏駅」となります。

（※大井町線において、自由が丘から大井町方面へ向かう際、各駅停車は「九品仏」に停まりますが、急行は通過して「大岡山」まで停まらないため、このような状況になります。）

Turn 2:
つばめちゃんが自由が丘駅から**溝の口方面（二子玉川・溝の口方面）**の急行に間違えて乗ってしまった場合を考えます。

1.  **自由が丘駅の次の急行停車駅：**
    大井町線の溝の口方面行き急行は、自由が丘の次は**「二子玉川」**に停車します。
2.  **目的地までの道のり：**
    つばめちゃんの目的地は、先ほどの問題から**「九品仏駅」**です。
    二子玉川駅で降りて、反対方向（自由が丘・大井町方面）の各駅停車に乗り換えます。
3.  **各駅停車の停車駅順：**
    二子玉川駅から自由が丘方面へ向かう各駅停車の停車駅は、以下の通りです。
    *   二子玉川駅（出発）
    *   → **尾山台駅**（1駅先）
    *   → **等々力駅**（2駅先）
    *   → **九品仏駅**（3駅先）

したがって、つばめちゃんは**3駅先**の駅で降りれば良いことになります。"""