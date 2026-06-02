import llm

theme = "時間"

prompt = f"""お題「{theme}」で川柳を10個作成してください。

川柳のルール：
- 5・7・5の17音で構成する
- ユーモアや風刺を交えた内容にする
- 日常的な場面や感情を詠む

出力形式：
番号. 川柳（読み仮名）
　　 一言コメント

10個すべて日本語で作成してください。"""

print(f"お題：{theme}\n")
result = llm.chat(
    prompt,
    system="あなたは川柳の名人です。ユーモアあふれる川柳を作ります。",
    temperature=0.9,
)
print(result)
