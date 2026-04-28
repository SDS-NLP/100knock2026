with open("popular-names.txt", encoding="utf-8") as f:
    texts      = f.readlines()
    tail_10    = texts[-10:]

print(tail_10)

# https://eng-entrance.com/linux-command-tail
# コマンド: tail -n 10 popular-names.txt