with open("popular-names.txt", encoding="utf-8") as f:
    texts      = f.readlines()
    head_10    = texts[:10]
print(head_10)

# https://eng-entrance.com/linux-command-head
# コマンド: head -n 10 popular-names.txt