import os
from google import genai
import pandas as pd
from knock02 import main

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

df = pd.read_csv("chapter05/philosophy.csv", header=None)


def move_correct_to_D(row):

    A = row[1]
    B = row[2]
    C = row[3]
    D = row[4]

    correct = row[5]

    if correct == "A":
        row[1] = D
        row[4] = A

    elif correct == "B":
        row[2] = D
        row[4] = B

    elif correct == "C":
        row[3] = D
        row[4] = C

    row[5] = "D"

    return row


if __name__ == "__main__":
    new_df = df.apply(move_correct_to_D, axis=1)

    new_df.to_csv("chapter05/philosophy_new.csv", header=False, index=False)

    filename = "chapter05/philosophy.csv"
    output = "chapter05/output_allD.txt"
    main(filename, output)
