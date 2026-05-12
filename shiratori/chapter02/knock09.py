filename = "shiratori/chapter02/popular-names.txt"

with open(filename, "r") as f:
    lines = f.readlines()


def get_third_col(line):
    cols = line.split()
    count = int(cols[2])
    return count


sorted_lines = sorted(lines, key=get_third_col, reverse=True)


with open("shiratori/chapter02/sorted09.txt", "w") as out:
    for line in sorted_lines:
        out.write(line)


# sort -k3,3nr shiratori/chapter02/popular-names.txt
