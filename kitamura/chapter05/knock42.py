from google import genai
import pandas as pd
import time 
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

csv = "japanese_idiom.csv"

df = pd.read_csv(csv, 
                 header=None,
                 names=['question', 'A','B','C','D','answer'])

correct_count = 0
total = len(df)

for index, row in tqdm(df.iterrows(), total=total):
    retry_count = 0
    while True:
        try:
            prompt = f"""
                    以下の質問に対して、最も適切な選択肢をA 、B 、C、Dの中から選び、その記号のみを答えなさい。
                    
                    質問: {row['question']}
                    A: {row['A']}
                    B: {row['B']}
                    C: {row['C']}
                    D: {row['D']}

                    解答:"""
            
            response = client.models.generate_content(
                            model="gemma-4-31b-it",
                            contents=prompt)
            correct_answer = row['answer']

            correct = (response.text == correct_answer)

            if correct:
                correct_count += 1

            else:
                print(f"{index+1}:{response.text}")

            time.sleep(5)
            break

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "ResourceExhausted" in error_msg:
                retry_count += 1
                # 指数的バックオフ（10秒, 20秒, 40秒, 80秒...と待機時間を増やす）
                wait_time = (2 ** retry_count) * 10 
                print(f"\n[待機] API制限。{wait_time}秒待機して再試行します... (Q{index+1})")
                time.sleep(wait_time)



accuracy = correct_count / total

print(f"正答率：{accuracy}")

# 107:C
# 正答率：0.9933333333333333