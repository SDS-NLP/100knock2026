import csv, re, time
from google import genai

client = genai.Client()

with open("data/jmmlu/abstract_algebra.csv", encoding="utf-8") as f:
    questions = list(csv.reader(f))[:5]

correct = 0
for row in questions:
    q, a, b, c, d, ans = row[:6]
    prompt = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n回答(A/B/C/Dの1文字):"
    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    pred = re.search(r"[ABCD]", res.text).group(0)
    print(f"予測={pred} 正解={ans}")
    if pred == ans.strip():
        correct += 1
    time.sleep(6)

print(f"正解率: {correct}/5")
