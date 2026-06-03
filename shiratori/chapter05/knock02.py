import os
import re
import time
import pandas as pd
from google import genai


def build_prompt(batch_df, start_idx):

    prompt = "Answer ONLY A, B, C, or D for each question.\n\n"

    for offset, row in enumerate(batch_df.itertuples(index=False)):

        q_num = start_idx + offset

        prompt += f"""
        Q{q_num}
        {row[0]}
        A: {row[1]}
        B: {row[2]}
        C: {row[3]}
        D: {row[4]}
        Answer format: Q{q_num}: A/B/C/D
"""

    return prompt


def main(filename, output):

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    df = pd.read_csv(filename, header=None)

    BATCH_SIZE = 30
    output_file = output

    with open(output_file, "w") as f:
        f.write("results\n\n")

    results = {}

    correct = 0
    total = 0

    for i in range(0, len(df), BATCH_SIZE):

        batch = df.iloc[i : i + BATCH_SIZE]

        prompt = build_prompt(batch, i)

        try:

            response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)

            text = response.text.strip()

            print(f"\n=== batch {i} ===")
            print(text)

            for idx, line in zip(batch.index, text.split("\n")):

                match = re.search(r"[ABCD]", line)

                if not match:
                    continue

                ans = match.group()

                results[f"Q{idx}"] = ans

                true_answer = df.iloc[idx, 5]

                total += 1

                if ans == true_answer:
                    correct += 1

        except Exception as e:
            print("ERROR:", e)

        time.sleep(5)

    accuracy = correct / total

    with open(output_file, "a") as f:

        for k, v in results.items():
            f.write(f"{k}: {v}\n")

        f.write("\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Correct: {correct}/{total}\n")

    print("\ndone")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Correct: {correct}/{total}")


if __name__ == "__main__":

    filename = "chapter05/philosophy.csv"
    output = "chapter05/output.txt"
    main(filename, output)
