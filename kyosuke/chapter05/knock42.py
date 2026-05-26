import os
import time
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

df = pd.read_csv("college_mathematics.csv", header=None)
df.columns = ["question", "A","B","C","D","answer"]
total_questions = len(df)
total_correct = 0

for index, row in df.iterrows():
    promopt = prompt = f"""
以下の多肢選択問題に解答し、最も適切な選択肢の記号（A, B, C, D）を1つだけ出力してください。
理由や解説は一切不要です。

問題: {row['question']}
A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}
"""
    answer = row['answer']
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,
            )
            
            ai_answer_raw = response.choices[0].message.content
            ai_answer = ai_answer_raw.strip()[0].upper()
            if ai_answer == answer:
                total_correct += 1
                result_mark = "⭕"
            else:
                result_mark = "❌"
                
            print(f"Q{index+1}: 正解={answer}, AI={ai_answer} -> {result_mark}")

            time.sleep(1.5)
            break
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate limit" in error_msg.lower():
                print(f"[制限到達] クォータ制限です。20秒間待機します... ({attempt+1}/{max_retries})")
                time.sleep(20)
            else:
                print(f"[エラー] サーバー通信エラーのため5秒待機します... ({attempt+1}/{max_retries})")
                time.sleep(5)

accuracy = total_correct / total_questions
print("-" * 30)
print(f"【最終結果】")
print(f"正解数: {total_correct} / {total_questions}")
print(f"正解率 (Accuracy): {accuracy * 100:.1f}%")
#35/99
#35.4%