#3列目の数値の逆順でファイルの各行を整列せよ（注意: 各行の内容は変更せずに並び替えよ）。同様の処理をsortコマンドで実現せよ。

def population(line):
    
    return int(line.split()[2])

with open("popular-names.txt", "r", encoding = "utf-8") as names:
    
    lines = names.readlines()
    
sorted_lines = sorted(lines, key = population, reverse = True)

for line in sorted_lines:
    
    print(line)

#sort -k3 -nr popular-names.txt    