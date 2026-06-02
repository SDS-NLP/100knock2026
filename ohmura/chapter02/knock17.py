# 17.1列目の文字列の異なり（文字列の種類）を求めよ。
# 確認にはcut, sort, uniqコマンドを用いよ。

file = "popular-names.txt"

with open(file, "r") as f:
    # 1列目（名前）だけをリストに取り出す
    names = [line.split("\t")[0] for line in f]

# set（集合）を使用することで、重複が自動的に消える
unique_names = set(names)

print(len(unique_names))

#cut -f 1 popular-names.txt | sort | uniq | wc -l