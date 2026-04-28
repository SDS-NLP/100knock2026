from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    print("".join(line.replace("\t", " ") for _, line in zip(range(10), f)))
