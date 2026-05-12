with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

ans_line = [line.rstrip("\n") for line in lines]
for i in range(10):
    print(ans_line[i])

"unixコマンドはhead popular-names.txt"