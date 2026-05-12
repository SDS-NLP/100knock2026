# 15. ファイルをN分割する

from math import ceil
from pathlib import Path


N = 10

base_dir = Path(__file__).parent
input_path = base_dir / "popular-names.txt"

with input_path.open("r", encoding="utf-8") as f:
    lines = f.readlines()

chunk_size = ceil(len(lines) / N)

for i in range(N):
    start = i * chunk_size
    end = start + chunk_size
    output_path = base_dir / f"popular-names_{i + 1}.txt"

    with output_path.open("w", encoding="utf-8") as f:
        f.writelines(lines[start:end])
