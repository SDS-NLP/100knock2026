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

senryu_list = """

"""
prompt = f"""次の川柳それぞれについて、面白さを10段階(1=つまらない, 10=非常に面白い)で評価してください。
各川柳に対して「番号: スコア - 理由」の形式で出力してください。

{senryu_list}"""

print(generate(prompt, temperature=0.0))