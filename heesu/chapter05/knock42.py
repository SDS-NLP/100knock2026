import csv
import re
import llm

def build_prompt(row: list[str]) -> str:
    question, a, b, c, d = row[0], row[1], row[2], row[3], row[4]
    return (
        f"{question}\n\n"
        f"A: {a}\n"
        f"B: {b}\n"
        f"C: {c}\n"
        f"D: {d}\n\n"
        "Answer with only the letter (A, B, C, or D)."
    )

def extract_answer(response: str) -> str:
    match = re.search(r"\b([A-D])\b", response.strip().upper())
    return match.group(1) if match else ""

def main():
    with open("college_physics.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    correct = 0
    total = len(rows)

    for i, row in enumerate(rows):
        prompt = build_prompt(row)
        response = llm.chat(prompt, temperature=0.0)
        predicted = extract_answer(response)
        label = row[5].strip().upper()
        is_correct = predicted == label
        correct += is_correct
        print(f"[{i+1}/{total}] predicted={predicted} correct={label} {'O' if is_correct else 'X'}")

    accuracy = correct / total * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.1f}%")

if __name__ == "__main__":
    main()
