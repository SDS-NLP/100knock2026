import csv
import urllib.request
from google import genai

client = genai.Client()

url = "https://raw.githubusercontent.com/nlp-waseda/JMMLU/main/JMMLU/machine_learning.csv"


response = urllib.request.urlopen(url)
lines = [line.decode('utf-8') for line in response.readlines()]
reader = csv.reader(lines)

next(reader)

correct = 0
total_questions = 5 

for i, row in enumerate(reader):
    if i >= total_questions:
        break
        
    q, a, b, c, d, ans = row[0], row[1], row[2], row[3], row[4], row[5]
    
    prompt = f"問題:{q}\nA:{a}\nB:{b}\nC:{c}\nD:{d}\n記号1文字（A/B/C/D）だけで答えてください。"
    res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)

    if res.text.strip().upper() == ans.strip().upper():
        correct += 1

print(f"正解率: {correct / total_questions * 100}%")