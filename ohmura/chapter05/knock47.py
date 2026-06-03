from google import genai

client = genai.Client()

senryu_data = """
1. 会議中 コード頭で 書いてるな
2. デプロイは 指先震え 祈るのみ
3. バグ見つけ 犯人自分と 知り愕然
4. 仕様変更 青天の霹靂 また書き直し
5. 納期前 眠気エナドリ 投入だ
6. コメントは 後で書くから 書かないまま
7. テスト書く 未来の自分に 任せよう
8. 固まった 再起動して また固まる
9. 腰痛い 椅子は投資と 決め購入
10. エラーログ 読めど読めども 宇宙語
"""

prompt = f"""
あなたは優秀な川柳の批評家です。
提示された10個の「エンジニア川柳」の面白さを、以下の基準で【10段階（1〜10）】で厳格に評価し、簡単な理由も添えて出力してください。
評価基準：共感性、ユーモア、五・七・五のリズム感

評価対象の川柳：
{senryu_data}

出力フォーマット：
番号. [スコア/10] 川柳
理由：〜〜
"""

try:
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )
    
    print("\n--- 評価結果 ---")
    print(response.text.strip())

except Exception as e:
    print(f"エラーが発生しました: {e}")