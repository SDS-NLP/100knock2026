with open("popular-names.txt", 'r') as f:
    lines = f.readlines()

sorted_lines = sorted(lines, key=lambda line: float(line.split()[2]), reverse=True)

for line in sorted_lines:
    print(line, end="")

"unixコマンドはsort -k 3 -n -r popular-names.txt"
