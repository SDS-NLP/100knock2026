import os
from google import genai

MODEL = 'gemini-3.1-flash-lite'
client = genai.Client(api_key=os.environ['API_KEY'])

TOPIC = '八王子'

gen_response = client.models.generate_content(
    model=MODEL,
    contents=f'お題「{TOPIC}」で川柳を10作品作ってください。番号なしで1行に1作品、川柳のみを出力してください。',
    config={'temperature': 1.0},
)
senryus = [s.strip() for s in gen_response.text.strip().split('\n') if s.strip()][:10]

print('生成した川柳:')
for i, s in enumerate(senryus, 1):
    print(f'{i}. {s}')

print('\n評価 (1〜10点):')
for i, senryu in enumerate(senryus, 1):
    eval_response = client.models.generate_content(
        model=MODEL,
        contents=(
            f'次の川柳の面白さを1〜10の整数で評価してください。数字のみ出力してください。\n\n'
            f'川柳: {senryu}\n\n評価:'
        ),
        config={'temperature': 0},
    )
    score = eval_response.text.strip()
    print(f'{i}. {senryu} → {score}点')

"""生成した川柳:
1. 高尾山登れば都心見下ろして
2. 八王子駅の改札迷路かな
3. いちょう並木秋には黄金色に染まり
4. 北口の賑わい過ぎて夜の街
5. 武蔵野の風も涼しき西の端
6. ラーメンの玉ねぎ甘し刻み込み
7. 学園の街路を歩く若者よ
8. 夕やけ小やけのメロディ響く駅
9. 八王子夏はあつくて冬寒し
10. 峠越え甲州街道歴史あり

評価 (1〜10点):
1. 高尾山登れば都心見下ろして → 3点
2. 八王子駅の改札迷路かな → 7点
3. いちょう並木秋には黄金色に染まり → 3点
4. 北口の賑わい過ぎて夜の街 → 4点
5. 武蔵野の風も涼しき西の端 → 5点
6. ラーメンの玉ねぎ甘し刻み込み → 5点
7. 学園の街路を歩く若者よ → 3点
8. 夕やけ小やけのメロディ響く駅 → 5点
9. 八王子夏はあつくて冬寒し → 3点
10. 峠越え甲州街道歴史あり → 3点"""