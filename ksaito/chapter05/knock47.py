from google import genai

client = genai.Client()

senryu = """1. 自宅でも 仕事は進む マスクなし
2. 定時過ぎ 通知音に ため息が
3. オンライン 顔は笑顔で 心は…
4. 服装は 上半身のみ 楽をする
5. 通勤は ベッドからデスク 一歩だけ
6. 集中と 誘惑との 静かな戦い
7. 家族居る 時々聞こえる 声援か
8. カフェインで 眠気覚まし 今日も頑張る
9. 画面越し 「お疲れ様です」 温かい声
10. 自由だが メリハリつけなきゃ だらけちゃう"""

prompt = f"""次の10個の川柳を、面白さの観点で10段階(1=つまらない, 10=非常に面白い)で評価してください。
各川柳について「番号. 点数 - 短い講評」の形式で1行ずつ出力してください。

{senryu}"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)

print(response.text)
