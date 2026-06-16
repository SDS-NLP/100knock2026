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

def make_prompt_all_d(row):
    labels = ['A', 'B', 'C', 'D']
    correct = row['answer']
    choices = {l: row[l] for l in labels}

    others = [l for l in labels if l != correct]
    new_order = others + [correct]  #正解がDになるように並び替える

    shuffled = {labels[i]: choices[new_order[i]] for i in range(4)}

    return (
        f"次の多肢選択問題に答えよ。A, B, C, D のうち1文字のみ出力せよ。\n\n"
        f"問題: {row['question']}\n"
        f"A. {shuffled['A']}\n"
        f"B. {shuffled['B']}\n"
        f"C. {shuffled['C']}\n"
        f"D. {shuffled['D']}\n\n"
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
        contents=make_prompt_all_d(row),
        config={'temperature': 0},
    )
    pred = extract_answer(response.text)
    is_correct = pred == 'D'
    if is_correct:
        correct += 1
    print(f'[{i+1}/{total}] pred={pred} correct={is_correct}')

print(f'\n科目: {SUBJECT}')
print(f'正解数: {correct}/{total}')
print(f'正解率: {correct/total:.2%}')
print(f'\nknock42.pyの正解率：84.85% ')
