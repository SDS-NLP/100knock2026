from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    print(sorted(set(line.split()[0] for line in f)))
