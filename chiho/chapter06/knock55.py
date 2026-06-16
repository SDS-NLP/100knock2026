"""55. アナロジーの正解率

54の実行結果を用いて、意味的アナロジーと文法的アナロジーの
正解率を測定する。
"""

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = SCRIPT_DIR / "knock54_results.txt"

SEMANTIC_SECTIONS = {
    "capital-common-countries",
    "capital-world",
    "currency",
    "city-in-state",
    "family",
}


def classify_section(section: str) -> str:
    """Return whether the section is semantic or syntactic."""
    if section in SEMANTIC_SECTIONS:
        return "semantic"
    return "syntactic"


def main() -> None:
    counts = {
        "semantic": {"correct": 0, "total": 0},
        "syntactic": {"correct": 0, "total": 0},
    }

    with RESULTS_PATH.open() as f:
        for line in f:
            stripped_line = line.strip()
            if not stripped_line:
                continue

            section, word1, word2, word3, word4, predicted_word, _similarity = stripped_line.split(
                "\t"
            )
            category = classify_section(section)
            counts[category]["total"] += 1
            if predicted_word == word4:
                counts[category]["correct"] += 1

    for category in ("semantic", "syntactic"):
        total = counts[category]["total"]
        correct = counts[category]["correct"]
        accuracy = correct / total if total else 0.0
        print(f"{category}: {accuracy:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
