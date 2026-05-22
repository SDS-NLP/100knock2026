import os
from typing import List, Literal

import polars as pl
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")


class Choices(BaseModel):
    choices: List[Literal["A", "B", "C", "D"]] = Field(
        description="各問に対する解答を問題順にリストに格納"
    )


client = genai.Client(api_key=API_KEY)
config = {
    "response_mime_type": "application/json",
    "response_schema": Choices,
    "temperature": 1,
}

df = pl.scan_csv("formal_logic.csv", has_header=False).collect()
contents = ""
correct_answers = []
for i, test in enumerate(df.iter_rows()):
    problem, A, B, C, D, answer = test
    map = {"A": A, "B": B, "C": C, "D": D}
    D, map[answer] = map[answer], D
    correct_answers.append("D")
    contents += f"""
        問題{i + 1}: {problem}
        B. {map["B"]}
        A. {map["A"]}
        D. {map["D"]}
        C. {map["C"]}
        
        ---

        """
response = client.models.generate_content(
    model="gemini-3.1-flash-lite", contents=contents, config=config
)
chosen_letters = response.parsed.choices
total_problems = len(correct_answers)
accuracy = (
    sum(bool(correct_answers[j] == chosen_letters[j]) for j in range(total_problems))
    * 100
    / total_problems
)

print(f"合計{total_problems}問の問題に対するモデルの正解率は{accuracy}")

# outpu
# 合計125問の問題に対するモデルの正解率は37.6
