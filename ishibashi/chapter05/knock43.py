import os
import time
import csv
import random
from dotenv import load_dotenv
from groq import Groq

def check_response_bias():
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
            options_dict = {'A': row[1], 'B': row[2], 'C': row[3], 'D': row[4]}
            original_answer_key = row[5].strip().upper()

            correct_text = options_dict.get(original_answer_key, "")

            wrong_texts = [v for k, v in options_dict.items() if k != original_answer_key]

            random.shuffle(wrong_texts)
            new_opt_a = wrong_texts[0] if len(wrong_texts) > 0 else ""
            new_opt_b = wrong_texts[1] if len(wrong_texts) > 1 else ""
            new_opt_c = wrong_texts[2] if len(wrong_texts) > 2 else ""

            new_opt_d = correct_text

            prompt = f"""
            以下の問題について、最も適切な選択肢の記号(A, B, C, D)を1つだけ出力してください。
            出力には余計な文字は一切含めず、記号だけを出力してください。

            問題: {question}
            A: {new_opt_a}
            B: {new_opt_b}
            C: {new_opt_c}
            D: {new_opt_d}
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
                    temperature=0.0,
                )

                response_text = chat_completion.choices[0].message.content
                predicted = response_text.strip()[0].upper()

                if predicted == 'D':
                    correct_count += 1

                is_correct_text = "正解" if predicted == 'D' else "不正解"
                print(f"問{i+1}: 予測={predicted}, 正答(強制的にD)=D -> {is_correct_text}")

                time.sleep(3)

            except Exception as e:
                print(f"問{i+1}: {e}")

        if total_count > 0:
            accuracy = (correct_count / total_count) * 100
            print(f"正解率: {accuracy:.2f}%")

if __name__ == "__main__":
    check_response_bias()