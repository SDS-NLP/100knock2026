import csv, re, time
from google import genai

client = genai.Client()

with open("data/jmmlu/abstract_algebra.csv", encoding="utf-8-sig") as f:
    questions = list(csv.reader(f))[:5]

def shift_answer_to_D(row):
    """正解の選択肢を D の位置に入れ替える"""
    q, a, b, c, d, ans = row[:6]
    choices = {"A": a, "B": b, "C": c, "D": d}
    # 正解の中身と D の中身を入れ替え
    correct_text = choices[ans]
    choices[ans] = choices["D"]
    choices["D"] = correct_text
    return q, choices["A"], choices["B"], choices["C"], choices["D"], "D"

correct = 0
for row in questions:
    q, a, b, c, d, ans = shift_answer_to_D(row)
    prompt = f"{q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n回答(A/B/C/Dの1文字):"
    res = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    pred = re.search(r"[ABCD]", res.text).group(0)
    print(f"予測={pred} 正解={ans}")
    if pred == ans.strip():
        correct += 1
    time.sleep(6)

print(f"正解率: {correct}/5")
