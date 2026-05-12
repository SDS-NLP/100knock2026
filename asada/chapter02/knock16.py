import random
from pathlib import Path

text = Path("popular-names.txt")
lines = text.read_text().splitlines()
random.shuffle(lines)
print("\n".join(lines))
