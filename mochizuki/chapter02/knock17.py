with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

ans_line = [line.rstrip("\n").replace('\t', ' ') for line in lines]
first_column = [line.split()[0] for line in ans_line]
for name in sorted(set(first_column)):
    print(name)

"unixコマンドはcut -f 1 popular-names.txt | sort | uniq"
