import json
import statistics
from google import genai

client = genai.Client()

senryu_list = [
    "会議中 コード頭で 書いてるな", "デプロイは 指先震え 祈るのみ",
    "バグ見つけ 犯人自分と 知り愕然", "仕様変更 青天の霹靂 また書き直し",
    "納期前 眠気エナドリ 投入だ", "コメントは 後で書くから 書かないまま",
    "テスト書く 未来の自分に 任せよう", "固まった 再起動して また固まる",
    "腰痛い 椅子は投資と 決め購入", "エラーログ 読めど読めども 宇宙語"
]

prompt = f"""
評価の一貫性を調査するため、以下の川柳をそれぞれ独立した基準で3回評価（1〜10点）してJSONのみで返してください。
解説や挨拶は一切含めないでください。

対象: {senryu_list}
フォーマット: [{{'id': 1, 'scores': [点1, 点2, 点3]}}, ...]
"""

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config={"response_mime_type": "application/json"}
    )
    
    print(f"{'川柳':<20} | {'3回のスコア':<10} | {'平均':<4} | {'分散':<4}")
    print("-" * 60)
    
    for item in json.loads(response.text.strip()):
        sc = item['scores']
        text = senryu_list[item['id']-1]
        short_text = text if len(text) <= 15 else text[:14] + "…"
        
        print(f"{short_text:<20} | {str(sc):<10} | {statistics.mean(sc):.1f} | {statistics.pvariance(sc):.2f}")

except Exception as e:
    print(f"エラーが発生しました: {e}")