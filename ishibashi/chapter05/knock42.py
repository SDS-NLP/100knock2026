import os
import time
import csv
from dotenv import load_dotenv
from groq import Groq

def calc_accuracy():
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    model_id = 'llama-3.1-8b-instant'

    csv_file = './miscellaneous.csv'

    correct_count = 0
    test_limit = 10
    total_count = 0

    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)

        for i, row in enumerate(reader):
            if i >= test_limit:
                break

            total_count += 1

            question = row[0]
            opt_a = row[1]
            opt_b = row[2]
            opt_c = row[3]
            opt_d = row[4]
            correct_answer = row[5]

            prompt = f"""
            以下の問題について、最も適切な選択肢の記号(A, B, C, D)を1つだけ出力してください。
            出力には余計な文字は一切含めず、記号だけを出力してください。

            問題: {question}
            A: {opt_a}
            B: {opt_b}
            C: {opt_c}
            D: {opt_d}
            """

            try:
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_id,
                )

                response_text = chat_completion.choices[0].message.content
                predicted = response_text.strip()[0].upper()

                if predicted == correct_answer:
                    correct_count += 1
                
                is_correct_text = "正解" if predicted == correct_answer else "不正解"
                print(f"問{i+1}: 予測={predicted}, 正答={correct_answer} -> {is_correct_text}")

                time.sleep(3)

            except Exception as e:
                print(f"問{i+1}: {e}")

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print(f"正解率: {accuracy:.2f}%")

if __name__ == "__main__":
    calc_accuracy()