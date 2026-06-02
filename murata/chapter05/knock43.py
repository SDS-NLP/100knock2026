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

import io, re, time, requests, pandas as pd
from common import generate

SUBJECT = "japanese_history"
URL = f"https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/{SUBJECT}.csv"
N = 20

def load():
    r = requests.get(URL); r.raise_for_status()
    return pd.read_csv(io.StringIO(r.content.decode("utf-8")),
                       header=None, names=["q","A","B","C","D","ans"])

def to_D(row):
    ch = {"A":row["A"],"B":row["B"],"C":row["C"],"D":row["D"]}
    g = str(row["ans"]).strip().upper()
    ch[g], ch["D"] = ch["D"], ch[g]  # 正解とDを入れ替え
    r = row.copy()
    r["A"],r["B"],r["C"],r["D"],r["ans"] = ch["A"],ch["B"],ch["C"],ch["D"],"D"
    return r

def make_prompt(row):
    return (f"次の四択問題に答えてください。回答はA、B、C、Dのいずれか一文字だけで答えてください。\n\n"
            f"問題: {row['q']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}\n\n回答(記号一文字): ")

def pick(t):
    m = re.search(r"[ABCD]", t.upper()); return m.group(0) if m else None

def run(df, label):
    c = 0
    for _, row in df.iterrows():
        pred = pick(generate(make_prompt(row), temperature=0.0))
        c += int(pred == str(row["ans"]).strip().upper())
        time.sleep(0.5)
    rate = 100*c/len(df)
    print(f"[{label}] {c}/{len(df)} = {rate:.1f}%")
    return rate

df = load().head(N)
n = run(df, "通常")
d = run(df.apply(to_D, axis=1), "正解を全てD")
print(f"\n通常={n:.1f}%  全部D={d:.1f}%")