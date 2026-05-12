import random

with open("popular-names.txt", encoding="utf-8") as f:
    lines = f.readlines()

random.shuffle(lines)
print("".join(lines), end="")

# 同様の処理は以下の shuf コマンドでも実現できます。
# gshuf popular-names.txt
# macOSではgshuf

