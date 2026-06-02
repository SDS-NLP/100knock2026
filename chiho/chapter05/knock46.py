# 46. 川柳の生成

"""
事前準備
python3 -m pip install google-genai
export GEMINI_API_KEY="your-api-key"
"""

import os

from google import genai
#from google.genai import errors, types

prompt = """
「5月」をテーマに，川柳を 10 個生成してください．
川柳は，５音・７音・５音で１つ得られる文章です．
1 ~ 10 までの番号と，川柳のみ出力してください．
"""

client = genai.Client()

response = client.models.generate_content_stream(
    model = "gemini-2.5-flash",
    contents = prompt
)

for chunk in response:
    print(chunk.text, end="", flush=True)

"""
実行例
1. 五月病 連休明けの 重い足
2. 大空に 夢を泳がす 鯉のぼり
3. 母の日や 普段言えない ありがとう
4. 風薫る 青葉まぶしい 五月晴れ
5. 柏餅 粒あん派だと 主張する
6. 連休は 過ぎてしまえば 夢の跡
7. 汗ばむ日 そろそろ準備 衣替え
8. 誇らしげ 居間に飾った 兜かな
9. 心地よい 薫風抜ける 散歩道
10. 連休で 財布の中身 五月晴れ
"""