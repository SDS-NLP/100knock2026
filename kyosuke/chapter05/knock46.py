import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

client = Groq(api_key=api_key)
prompt = """
「学習」をお題に川柳を10個生成せよ
ただし必ず5音-7音-5音で構成せよ
"""
print("川柳を生成中...\n")

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
    print(response.choices[0].message.content)
except Exception as e:
    print(f"エラーが発生しました: {e}")