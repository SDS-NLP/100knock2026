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
これらは、お題『ミュトスの侵略』に基づいた10個の川柳です。
これらの川柳の面白さを10段階で評価してください。（1が最も面白くない、10が最も面白い）

1個目の川柳: 異形の影 迫りくる脅威 眠れぬ夜
2個目の川柳: 響く咆哮 絶望の幕開け 武器を握れ
3個目の川柳: 奪われる故郷 故郷の緑 涙あふれる
4個目の川柳: 団結の光 勇気は希望 立ち向かう者
5個目の川柳: 科学の力 知恵を絞って 活路を見出す
6個目の川柳: 仲間と共に 守るべきもの 誓う未来
7個目の川柳: 巨獣の爪 街を裂く叫び 響き渡る
8個目の川柳: 犠牲払って 掴んだ勝利 鎮魂の歌
9個目の川柳: 静寂戻る 戦いの後 傷跡残る
10個目の川柳: 未来のため 語り継ぐ歌 侵略の証

以下のような形式で回答してください。
1個目の川柳の評価: 
2個目の川柳の評価:
...
"""

response = model.generate_content(prompt)
print(response.text)