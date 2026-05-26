import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

prompt = """
お題を「アルバイト」として、川柳の案を10個作成せよ。
"""

response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
)

print(response.text)
