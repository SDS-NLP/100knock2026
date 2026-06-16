import os
from google import genai

MODEL = 'gemini-3.1-flash-lite'
client = genai.Client(api_key=os.environ['API_KEY'])

TOPIC = '八王子'

prompt = f"""お題「{TOPIC}」で川柳を10作品作ってください。
川柳は「五・七・五」の音節構造にしてください。
番号付きリスト形式で出力してください。"""

response = client.models.generate_content(
    model=MODEL,
    contents=prompt,
    config={'temperature': 1.0},
)
print(f'お題: {TOPIC}')
print(response.text)

"""お題: 八王子
生成した川柳:
1. 高尾山登れば都心見下ろして
2. 八王子駅の改札迷路かな
3. いちょう並木秋には黄金色に染まり
4. 北口の賑わい過ぎて夜の街
5. 武蔵野の風も涼しき西の端
6. ラーメンの玉ねぎ甘し刻み込み
7. 学園の街路を歩く若者よ
8. 夕やけ小やけのメロディ響く駅
9. 八王子夏はあつくて冬寒し
10. 峠越え甲州街道歴史あり"""