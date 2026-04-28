with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

ans_line = [line.rstrip("\n").replace('\t', ' ') for line in lines]
first_column = [line.split()[0] for line in ans_line]

count_dict = {}
for name in first_column:
    if name in count_dict:
        count_dict[name] += 1
    else:
        count_dict[name] = 1

for name, count in sorted(count_dict.items(), key=lambda x: x[1], reverse=True):
    print(count, name)

"unixコマンドはcut -f 1 popular-names.txt | sort | uniq -c | sort -r"
