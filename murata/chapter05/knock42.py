import io, re, time, requests, pandas as pd
SUBJECT = "japanese_history"
URL = f"https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/{SUBJECT}.csv"
N=20
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

def load():
    r = requests.get(URL)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8")),
                       header=None, names=["q","A", "B", "C", "D", "ans"])

def make_prompt(row):
    return (f"次の四択問題に答えてください。回答はA、B、C、Dのいずれか一文字だけで答えてください\n\n"
            f"問題: {row['q']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n\n回答(記号一文字): ")

def pick(text):
    m = re.search(r"[ABCD]", text.upper())
    return m.group(0) if m else None

df = load().head(N)
correct = 0
for i, row in df.iterrows():
    pred = pick(generate(make_prompt(row), temperature=0.0))
    gold = str(row["ans"]).strip().upper()
    correct += int(pred == gold)
    time.sleep(0.5)
print(f"正解率: {correct}/{len(df)} = {100 * correct / len(df):.1f}%")