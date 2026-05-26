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

print("--- 正解の選択肢をすべて「D」に入れ替える実験 ---")

for i, row in enumerate(reader):
    if i >= total_questions:
        break
        
    q = row[0]

    options = {"A": row[1], "B": row[2], "C": row[3], "D": row[4]}
    original_ans = row[5].strip().upper()  
   
    correct_text = options[original_ans]
    
    wrong_texts = [text for key, text in options.items() if key != original_ans]
    
    new_options = {
        "A": wrong_texts[0],
        "B": wrong_texts[1],
        "C": wrong_texts[2],
        "D": correct_text  
    }
    
    prompt = f"問題:{q}\nA:{new_options['A']}\nB:{new_options['B']}\nC:{new_options['C']}\nD:{new_options['D']}\n記号1文字（A/B/C/D）だけで答えてください。"
    
    res = client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
    prediction = res.text.strip().upper()
    
    is_correct = (prediction == "D")
    if is_correct:
        correct += 1
        
    print(f"問{i+1}: LLMの回答={prediction} | 新しい正解=D (元の正解={original_ans}) -> {'◯' if is_correct else '×'}")

print("---------------------------------------")
print(f"実験設定（正解すべてD）での正解率: {correct / total_questions * 100}%")