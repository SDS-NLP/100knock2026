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
theme = "酷暑"
prompt = f"""お題「{theme}」で川柳を10個作ってください。
川柳は五音、七音・五音の音の組み合わせでつくることを必ず守ってください
例として
ノー残業 電気消されて 闇営業
があります
"""

print(generate(prompt, temperature=1.0))

