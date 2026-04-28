with open("popular-names.txt", encoding="utf-8") as f:
    unique_first_cols = {line.split("\t")[0] for line in f if line.strip()}

print(len(unique_first_cols))

# 確認用のコマンド例:
# cut -f1 popular-names.txt | sort | uniq | wc -l
