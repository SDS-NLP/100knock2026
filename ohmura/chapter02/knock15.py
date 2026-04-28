import math

filename = "popular-names.txt"
N = 10

with open("popular-names.txt", "r") as f:
    lines = f.readlines()
    total_lines = len(lines)

unit = math.ceil(total_lines / N)

for i in range(N):
    start = i * unit
    end = (i + 1) * unit

    with open(f"child_{i:02d}.txt", "w", encoding="utf-8") as out_f:
        out_f.writelines(lines[start:end])

print(f"全 {total_lines} 行を {N} 個のファイルに分割しました。")