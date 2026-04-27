from pathlib import Path

text = Path("popular-names.txt")

with text.open() as f:
    count = sum(1 for line in f)

print(f"ファイルの行数: {count}")
