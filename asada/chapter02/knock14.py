from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    print("\n".join(line.split()[0] for _, line in zip(range(10), f)))
