import warnings
warnings.filterwarnings("ignore")

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Gemini APIの設定
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash-lite")

# 問題文
prompt = """
次のお題に関する川柳を10個作成してください。
お題:ミュトスの侵略

以下のような形式で回答してください。
1個目の川柳: 
2個目の川柳:
...
"""

response = model.generate_content(prompt)
print(response.text)