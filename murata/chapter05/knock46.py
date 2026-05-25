import os 
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemini-2.5-flash"
_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

def generate(prompt, temperature=None):
    cfg = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
    res = _client.models.generate_content(model=MODEL, contents=prompt, config=cfg)
    return res.text
theme = "酷暑"
prompt = f"""お題「{theme}」で川柳を10個作ってください。
川柳は五・七・五の音数で、ユーモアやあるあるネタを含めてください。
1行に1つ、番号を付けて出力してください。"""

print(generate(prompt, temperature=1.0))

