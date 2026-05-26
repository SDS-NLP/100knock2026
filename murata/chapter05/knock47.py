import os 
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.environ.get("LLM_BACKEND", "ollama") 


import ollama
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
_client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

def generate(prompt, temperature=None):
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    res = _client.generate(model=MODEL, prompt=prompt, options=options)
    return res["response"]
senryu_list = """
1.  ドラゴン　大陸　海から雨が降りて
2.  おかえどさん　山中　草木　昔よりすばらしく成長した
3.  おととし　夏　に雨が降って
4.  てつはん　熱帯　風が吹いて
5.  にせの　なりきり　太陽から影をもらって
6.  さくら　草木　雪が終わって
7.  あたま　熱中症　水分を多くする必要がある
8.  はかれさん　大海　夏は休む
9.  おいしい　にんじん　夏が終わるまで
10. にごりおし　雨天　お花見
"""
prompt = f"""次の川柳それぞれについて、面白さを10段階(1=つまらない, 10=非常に面白い)で評価してください。
各川柳に対して「番号: スコア - 理由」の形式で出力してください。

{senryu_list}"""

print(generate(prompt, temperature=0.0))