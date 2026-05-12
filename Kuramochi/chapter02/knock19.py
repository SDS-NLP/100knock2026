with open("popular-names.txt", encoding="utf-8") as f:
    lines = [line.rstrip("\n") for line in f if line.strip()]

lines.sort(key=lambda line: float(line.split("\t")[2]), reverse=True)
print("\n".join(lines))

# 同様の確認用コマンド例:
# sort -k3,3nr popular-names.txt
