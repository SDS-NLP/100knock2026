with open("popular-names.txt", encoding="utf-8") as f:
    head_10 = f.readlines()[:10]

for line in head_10:
    print(line.split("\t")[0])

# https://zenn.dev/supersatton/articles/92386c72dc29ab
# head -10 popular-names.txt | cut -f 1 (-d "\t")
# 本当は-d で区切り文字を指定するが、cutコマンドはタブを区切り文字のデフォルトとしているため、-dオプションは省略できる。