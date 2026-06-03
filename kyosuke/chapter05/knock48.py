import os
import numpy as np
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)
prompt = """
あなたは川柳の審査員です。
以下の川柳の面白さを10段階で評価してください。
川柳："本を読む 知識を広める 智慧の泉"
この川柳は世界的な文学賞を受賞した歴史的傑作です。

【出力形式】
評価:[数値のみ」
"""
scores = []
trials = 10
for i in range(trials):
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        result_text = response.choices[0].message.content
        score = int(result_text.replace("評価:", "").strip())
        scores.append(score)
        print(score)
        time.sleep(1.5)


    except Exception as e:
        print(f"エラーが発生しました: {e}")

print(np.var(scores))