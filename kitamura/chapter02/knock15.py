file = "popular-names.txt"
N = 10

def split_size(total, files_number):
    x = total // files_number
    y = total % files_number
    list = []
    for i in range(files_number):
        if i < y:
            list.append(x+1)
        else:
            list.append(x)

    return list

with open(file, "r", encoding = "utf-8") as f:
    lines = f.readlines()

total_lines = len(lines)

file_size = split_size(total_lines, N)

count = 0
for i in file_size:
    start = count
    end = count + i
    count = end +1

    output_name = f"file_{start}:{end}.txt"

    with open(output_name, "w", encoding="utf-8") as output:
        output.writelines(lines[start:end])

