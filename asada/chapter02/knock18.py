from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    names = [line.split()[0] for line in f]
freq = {name: names.count(name) for name in set(names)}
print(sorted(freq.items(), key=lambda item: item[1], reverse=True))
