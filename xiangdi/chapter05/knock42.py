import os
import re
import time
from datasets import load_dataset
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

subject = "philosophy"

dataset = load_dataset("nlp-waseda/JMMLU", subject, split="test", trust_remote_code=True,)

num_questions = 5
dataset = dataset.select(range(num_questions))

def make_prompt(example):
    return f"""
以下の多肢選択問題に答えよ。
選択肢 A, B, C, D のうち、正解の記号のみを出力せよ。

問題:
{example["question"]}

A. {example["A"]}
B. {example["B"]}
C. {example["C"]}
D. {example["D"]}

正解:
"""

def extract_answer(text):
    text = text.strip()

    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)

    return None

correct = 0

for i, example in enumerate(dataset):
    prompt = make_prompt(example)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
    )

    output = response.text
    pred = extract_answer(output)
    ans = example["answer"]

    is_correct = pred == ans
    if is_correct:
        correct += 1

    time.sleep(1)

accuracy = correct / num_questions

print()
print(f"科目: {subject}")
print(f"問題数: {num_questions}")
print(f"正解率: {accuracy:.2%}")