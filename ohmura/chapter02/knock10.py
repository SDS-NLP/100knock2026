filename = "popular-names.txt"

count = 0

with open("popular-names.txt", "r") as f:
    for line in f:
        count += 1

print(f"ファイルの行数は {count} 行です。")