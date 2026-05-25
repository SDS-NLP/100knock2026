import os
import re
import pandas as pd
from google import genai

SUBJECT = 'computer_security'
CSV_PATH = f'{SUBJECT}.csv'
MODEL = 'gemini-3.1-flash-lite'

client = genai.Client(api_key=os.environ['API_KEY'])

df = pd.read_csv(CSV_PATH, header=None,
                 names=['question', 'A', 'B', 'C', 'D', 'answer'])

def make_prompt(row):
    return (
        f"次の多肢選択問題に答えよ。A, B, C, D のうち1文字のみ出力せよ。\n\n"
        f"問題: {row['question']}\n"
        f"A. {row['A']}\n"
        f"B. {row['B']}\n"
        f"C. {row['C']}\n"
        f"D. {row['D']}\n\n"
        f"答え:"
    )

def extract_answer(text):
    if not text:
        return None
    m = re.search(r'\b([ABCD])\b', text.strip().upper())
    return m.group(1) if m else None

correct = 0
total = len(df)

for i, row in df.iterrows():
    response = client.models.generate_content(
        model=MODEL,
        contents=make_prompt(row),
        config={'temperature': 0},
    )
    pred = extract_answer(response.text)
    gold = row['answer']
    is_correct = pred == gold
    if is_correct:
        correct += 1
    print(f'[{i+1}/{total}] pred={pred} gold={gold} correct={is_correct}')

print(f'\n科目: {SUBJECT}')
print(f'正解数: {correct}/{total}')
print(f'正解率: {correct/total:.2%}')

"""
科目: computer_security
正解数: 84/99
正解率: 84.85%
"""