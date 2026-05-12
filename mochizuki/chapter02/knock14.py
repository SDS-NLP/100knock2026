with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

ans_line = [line.rstrip("\n").replace('\t', ' ') for line in lines]
for i in range(10):
    print(ans_line[i].split()[0])

"unixコマンドはhead -n 10 popular-names.txt | cut -f 1"