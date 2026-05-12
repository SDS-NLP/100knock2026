import math

def split_file_n(filename, n):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    lines_per_file = math.ceil(total_lines / n)
    
    for i in range(n):
        part = lines[i * lines_per_file : (i + 1) * lines_per_file]
        if not part:
            break
        with open(f"split_{i+1}.txt", 'w', encoding='utf-8') as f:
            f.writelines(part)

split_file_n("popular-names.txt", 10)


"unixコマンドはsplit -l 278 popular-names.txt split_"