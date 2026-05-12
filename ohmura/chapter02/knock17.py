filename = "popular-names.txt"

with open(filename, "r", encoding="utf-8") as f:
    # 1列目（名前）だけをリストに取り出す
    names = [line.split("\t")[0] for line in f]

# set（集合）に変換することで、重複が自動的に消える
unique_names = set(names)

# 種類の数と、中身の確認
print(f"一列目の文字列の種類は全部で {len(unique_names)} 種類です。")