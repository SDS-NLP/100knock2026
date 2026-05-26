import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)
senryu_list = [
    "学ぶことよい 努力が大切だ 未来のために",
    "頭に詰まる 勉強は楽しい 知恵の宝石",
    "本を読む 知識を広める 智慧の泉",
    "教室で学ぶ 友達と一緒に 知恵の共有",
    "学び続けよ 未来が明るく 輝く道へ",
    "勉強は苦よ しかし知恵が 未来を拓く",
    "知識を深め 自信を高める 人生の基礎",
    "学ぶ心よ 常に新しい 発見の喜び",
    "教育の力 人生を変える 希望の光よ",
    "未来のために 学び続けること 自分を高める"
]
prompt_template = """
あなたは川柳の審査員です。
以下の川柳の面白さを10段階で評価してください。
川柳：{senryu}

「出力形式」
川柳：....
評価: X/10」

"""

for senryu in senryu_list:
    prompt = prompt_template.format(senryu=senryu)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"エラーが発生しました: {e}")

